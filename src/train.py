import torch
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T
from torchvision.models import resnet50, ResNet50_Weights
import os
import wandb
import datetime
from utils import (
    EarlyStopping,
    get_all_identities,
    get_portion_of_identities,
    evaluate_model,
    build_label_to_images,
    ensure_dir,
    set_seed,
    extract_embeddings,
    plot_and_log_tsne,
    train_one_epoch,
    validate,
    build_label_to_imgs,
    random_subset_label_dict,
)
from model import FaceVerificationModel
from dataset import TSNEDataset, FaceVerificationDataset
from loss import hybrid_triplet_loss
import config

set_seed(config.SEED)

for _dir in (config.MODELS_DIR, config.TSNE_PLOTS_DIR, config.ROC_PLOTS_DIR):
    ensure_dir(_dir)

print("Directories created or already exist!")

BATCH_SIZE = config.BATCH_SIZE
LR = config.LR
DECAY = config.DECAY
DROPOUT = config.DROPOUT
EPOCHS = config.EPOCHS
FACTOR = config.FACTOR
EMB_DIM = config.EMB_DIM
THRESHOLD = config.THRESHOLD
ALPHA = config.ALPHA
N_EPOCHS_MARGIN = config.N_EPOCHS_MARGIN
MARGIN = config.MARGIN
INITIAL_MARGIN = config.INITIAL_MARGIN
PEAK_MARGIN = config.PEAK_MARGIN
FINAL_MARGIN = config.FINAL_MARGIN

now = datetime.datetime.now()

name = now.strftime("experiment_%d_%m_%H_%M")

wandb.init(
    project=config.WANDB_PROJECT,
    name=name,
    mode=config.WANDB_MODE,
    config={
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "architecture": "resnet50",
        "epochs": EPOCHS,
        "loss": "hybrid triplet loss",
        "alpha": ALPHA,
        "lr_scheduler": "ReduceOnPlateau",
        "lr factor": FACTOR,
        "margin": MARGIN,
        "n_epochs_for_margin": N_EPOCHS_MARGIN,
        "dropout": DROPOUT,
        "dataset": "fullvggface2",
        "emb_dim": EMB_DIM,
        "threshold": THRESHOLD,
        "augmentations": "agressive",
        "margin_schedueler": True,
        "cosine_sim_loss": True,
        "hard negative mining": True,
        "initial_margin": INITIAL_MARGIN,
        "peak_margin": PEAK_MARGIN,
        "final_margin": FINAL_MARGIN,
    },
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transforms = T.Compose(
    [
        T.Resize((config.TRAIN_RESIZE, config.TRAIN_RESIZE)),
        T.RandomCrop(config.INPUT_SIZE),  # Random crop
        T.RandomHorizontalFlip(),
        T.RandomRotation(config.TRAIN_ROTATION_DEG),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.RandomAffine(
            degrees=config.TRAIN_AFFINE_DEG, translate=(0.05, 0.05), scale=(0.95, 1.05)
        ),  # More conservative
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_transforms = T.Compose(
    [
        T.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

train_data_path = config.TRAIN_DATA_DIR
test_data_path = config.VAL_DATA_DIR

for _d in (train_data_path, test_data_path):
    if not os.path.isdir(_d):
        raise SystemExit(
            f"Dataset not found: {_d} does not exist. "
            "Place identity folders of face images under data/train/ and "
            "data/val/ (or set DATA_DIR). See the README."
        )

train_identities = get_all_identities(train_data_path)

train_chosen_identities = get_portion_of_identities(
    train_identities, config.TRAIN_IDENTITY_FRACTION
)
train_label_to_imgs = build_label_to_images(train_data_path, train_chosen_identities)

test_identities = get_all_identities(test_data_path)
test_label_to_imgs = build_label_to_images(test_data_path, test_identities)

train_dataset = FaceVerificationDataset(train_label_to_imgs, train_transforms)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=config.NUM_WORKERS,
    persistent_workers=True,
)

test_dataset = FaceVerificationDataset(test_label_to_imgs, test_transforms)
test_loader = DataLoader(
test_dataset,
batch_size=BATCH_SIZE,
num_workers=config.NUM_WORKERS,
persistent_workers=True,
)

backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
model = FaceVerificationModel(backbone, dropout=DROPOUT, embedding_size=EMB_DIM).to(
    device
)
# The compiled module is only used for the training forwards; saves/reloads use
# the original model so checkpoints have plain (unprefixed) state_dict keys.
compiled_model = torch.compile(model, mode="default")

loss_fn = hybrid_triplet_loss
early_stopping = EarlyStopping(
    patience=config.EARLY_STOP_PATIENCE,
    min_delta=config.EARLY_STOP_MIN_DELTA,
    path=config.MODEL_PATH,
)

optimizer = torch.optim.AdamW(
    [
        {"params": model.backbone.layer4.parameters(), "lr": config.LAYER4_LR},
        {"params": model.embedding_layer.parameters(), "lr": LR},
    ],
    weight_decay=DECAY,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=FACTOR, patience=3
)
scaler = torch.amp.GradScaler("cuda")

for epoch in range(EPOCHS):
    train_loss = train_one_epoch(
        compiled_model, train_loader, optimizer, loss_fn, scaler, device, epoch
    )
    val_loss, val_acc = validate(
        compiled_model, test_loader, loss_fn, device, epoch, threshold=THRESHOLD
    )

    if epoch > config.EARLY_STOP_WARMUP:
        early_stopping(val_loss, val_acc, model, optimizer)
    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[-1]["lr"]

    wandb.log(
        {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch": epoch,
            "learning_rate": current_lr,
        }
    )

    if early_stopping.early_stop:
        print("Early stopping triggered! Stopping training.")
        break

wandb.log({"best_loss": early_stopping.best_loss})

if not os.path.exists(early_stopping.path):
    # Short run (EPOCHS <= EARLY_STOP_WARMUP): early stopping never fired, so
    # the model was never saved — save the final model instead of crashing.
    torch.save(model.state_dict(), early_stopping.path)
    print("saved final model (early stopping never triggered)")

root_dir = config.VAL_DATA_DIR

label_to_imgs = build_label_to_imgs(root_dir)

subset_dict = random_subset_label_dict(
    label_to_imgs, num_identities=10, max_imgs_per_identity=20
)

tsne_dataset = TSNEDataset(subset_dict, transform=test_transforms)
model.load_state_dict(
    torch.load(config.MODEL_PATH, map_location=device)
)

tsne_dataloader = DataLoader(
    tsne_dataset, batch_size=config.EVAL_BATCH_SIZE, shuffle=False
)

embeddings, labels = extract_embeddings(model, tsne_dataloader, device)

plot_and_log_tsne(
    embeddings,
    labels,
    name=name,
    perplexity=config.TSNE_PERPLEXITY,
    artifact_name="tsne_" + name,
)

evaluate_model(model, device, root_dir, test_transforms, name=name)
