import torch
import torch.nn.functional as F
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

set_seed(42)

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)  # Go one level up
os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "tsneplots"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "ROCplots"), exist_ok=True)

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
    project="face-verification",
    name=name,
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
        T.Resize((256, 256)),
        T.RandomCrop(224),  # Random crop
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.RandomAffine(
            degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)
        ),  # More conservative
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_transforms = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

dataset_path = "./data"

train_data_path = os.path.join(dataset_path, "train")
test_data_path = os.path.join(dataset_path, "val")

train_identities = get_all_identities(train_data_path)

train_chosen_identities = get_portion_of_identities(train_identities, 1)
train_label_to_imgs = build_label_to_images(train_data_path, train_chosen_identities)

test_identities = get_all_identities(test_data_path)
test_label_to_imgs = build_label_to_images(test_data_path, test_identities)

train_dataset = FaceVerificationDataset(train_label_to_imgs, train_transforms)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=min(4, os.cpu_count() // 2),
    persistent_workers=True,
)

test_dataset = FaceVerificationDataset(test_label_to_imgs, test_transforms)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    num_workers=min(4, os.cpu_count() // 2),
    persistent_workers=True,
)

backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
model = FaceVerificationModel(backbone, dropout=DROPOUT, embedding_size=EMB_DIM).to(
    device
)
model = torch.compile(model, mode="default")

loss_fn = hybrid_triplet_loss
early_stopping = EarlyStopping(
    patience=5, min_delta=0.001, path="./models/" + name + "_model.pth"
)

optimizer = torch.optim.AdamW(
    [
        {"params": model.backbone.layer4.parameters(), "lr": 5e-5},
        {"params": model.embedding_layer.parameters(), "lr": LR},
    ],
    weight_decay=DECAY,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=FACTOR, patience=3
)
scaler = torch.cuda.amp.GradScaler()

for epoch in range(EPOCHS):
    train_loss = train_one_epoch(
        model, train_loader, optimizer, loss_fn, scaler, device, epoch
    )
    val_loss, val_acc = validate(
        model, test_loader, loss_fn, device, epoch, threshold=THRESHOLD
    )

    if epoch > 10:
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

root_dir = "./data/val/"

label_to_imgs = build_label_to_imgs(root_dir)

subset_dict = random_subset_label_dict(
    label_to_imgs, num_identities=10, max_imgs_per_identity=20
)

tsne_dataset = TSNEDataset(subset_dict, transform=test_transforms)
model.load_state_dict(
    torch.load("./models/" + name + "_model.pth", map_location=device)
)

tsne_dataloader = DataLoader(tsne_dataset, batch_size=32, shuffle=False)

embeddings, labels = extract_embeddings(model, tsne_dataloader, device)

plot_and_log_tsne(
    embeddings, labels, name=name, perplexity=40, artifact_name="tsne_" + name
)

evaluate_model(model, device, root_dir, test_transforms, name=name)
