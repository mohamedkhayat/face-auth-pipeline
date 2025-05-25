import os
import random
import torch
from PIL import Image
import numpy as np
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import wandb
import config
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
)
from dataset import TSNEDataset
from torch.utils.data import DataLoader


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_all_identities(root_dir):
    all_entries = os.listdir(root_dir)
    identities = []
    for entry in all_entries:
        full_path = os.path.join(root_dir, entry)
        if os.path.isdir(full_path):
            identities.append(entry)
    return identities


def get_portion_of_identities(identities, fraction):
    n_identities = len(identities)

    random.shuffle(identities)
    cutoff = int(n_identities * fraction)
    chosen_identities = identities[:cutoff]
    return chosen_identities


def build_label_to_images(root_dir, chosen_identities):
    label_to_images = {}

    for label in chosen_identities:
        label_path = os.path.join(root_dir, label)
        if os.path.isdir(label_path):
            image_names = os.listdir(label_path)

            image_paths = [
                os.path.join(label_path, image_name) for image_name in image_names
            ]

            if len(image_paths) >= 2:
                label_to_images[label] = image_paths

    return label_to_images


def get_embedding(image_path, model, transform, device):
    """
    Load an image, apply transforms, and compute its embedding.
    """
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(image)
    return embedding.squeeze().cpu().numpy()


def margin_schedule(
    epoch,
    initial_margin=config.INITIAL_MARGIN,
    peak_margin=config.PEAK_MARGIN,
    final_margin=config.FINAL_MARGIN,
    peak_epoch=15,
    total_epochs=config.N_EPOCHS_MARGIN,
):
    if epoch < peak_epoch:
        return initial_margin + (peak_margin - initial_margin) * (
            epoch / peak_epoch
        )  # Gradual increase
    else:
        return peak_margin - (
            (peak_margin - final_margin)
            * ((epoch - peak_epoch) / (total_epochs - peak_epoch))
        )


class EarlyStopping:
    def __init__(
        self, patience=5, min_delta=0.0, path="./models/earlystopping_model.pth"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, val_acc, model, optimizer):
        if self.best_loss is None:
            self.best_loss = val_loss
            torch.save(model.state_dict(), self.path)
            print("saved model")

        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0

            torch.save(model.state_dict(), self.path)
            print(f"saved model at loss : {val_loss:.4f} - Accuracy: {val_acc:.4f}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def train_one_epoch(
    model, loader, optimizer, loss_fn, scaler, device, epoch, margin_sched=True
):
    model.train()
    running_loss = 0.0

    for step, (anchor_imgs, pos_imgs, neg_imgs) in enumerate(loader):
        anchor_imgs = anchor_imgs.to(device)
        pos_imgs = pos_imgs.to(device)
        neg_imgs = neg_imgs.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            anchor_embs = model(anchor_imgs)
            pos_embs = model(pos_imgs)
            neg_embs = model(neg_imgs)
            loss = loss_fn(anchor_embs, pos_embs, neg_embs, epoch)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    return running_loss / len(loader)


@torch.no_grad()
def validate(model, loader, loss_fn, device, epoch, threshold=0.75, margin_sched=True):
    model.eval()
    running_loss = 0.0
    all_sim = []
    all_labels = []

    for anchor_imgs, pos_imgs, neg_imgs in loader:
        anchor_imgs = anchor_imgs.to(device)
        pos_imgs = pos_imgs.to(device)
        neg_imgs = neg_imgs.to(device)

        anchor_embs = model(anchor_imgs)
        pos_embs = model(pos_imgs)
        neg_embs = model(neg_imgs)

        batch_loss = loss_fn(anchor_embs, pos_embs, neg_embs, epoch, margin_sched)
        running_loss += batch_loss.item()

        pos_sim = F.cosine_similarity(anchor_embs, pos_embs)
        neg_sim = F.cosine_similarity(anchor_embs, neg_embs)
        all_sim.extend(pos_sim.cpu().numpy())
        all_labels.extend([1] * len(pos_sim))
        all_sim.extend(neg_sim.cpu().numpy())
        all_labels.extend([0] * len(neg_sim))

    all_sim = np.array(all_sim)
    all_labels = np.array(all_labels)
    best_acc = np.mean((all_sim > threshold) == all_labels)

    return running_loss / len(loader), best_acc


def extract_embeddings(model, dataloader, device):
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            embeddings = model(images)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)

    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    import itertools

    all_labels = list(itertools.chain(*all_labels))
    all_labels = np.array(all_labels)
    return all_embeddings, all_labels


def plot_and_log_tsne(
    embeddings, labels, name, perplexity=30, artifact_name="tsne_plots"
):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    reduced = tsne.fit_transform(embeddings)

    unique_labels = np.unique(labels)
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    numeric_labels = np.array([label_to_index[label] for label in labels])

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        reduced[:, 0], reduced[:, 1], c=numeric_labels, cmap="viridis", alpha=0.7
    )
    cbar = plt.colorbar(scatter, ticks=range(len(unique_labels)))
    cbar.ax.set_yticklabels(unique_labels)

    plt.title("t-SNE Visualization of Face Embeddings")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")

    plot_filename = "./tsneplots/tsne_plot" + name + ".png"
    plt.savefig(plot_filename)
    plt.close()

    artifact = wandb.Artifact(artifact_name, type="analysis")
    artifact.add_file(plot_filename)

    wandb.log_artifact(artifact)

    wandb.log({"tsne_plot": wandb.Image(plot_filename)})


def build_label_to_imgs(root_dir):
    """
    root_dir: path to the directory containing subfolders like 'n000001', 'n000009', etc.

    Returns a dictionary: { 'n000001': [path1, path2, ...], 'n000009': [...], ... }
    """
    label_to_imgs = {}
    for identity_folder in os.listdir(root_dir):
        identity_path = os.path.join(root_dir, identity_folder)

        if os.path.isdir(identity_path):
            image_filenames = os.listdir(identity_path)

            image_paths = [
                os.path.join(identity_path, img_name)
                for img_name in image_filenames
                if img_name.lower().endswith((".jpg", ".png", ".jpeg"))
            ]

            if len(image_paths) > 0:
                label_to_imgs[identity_folder] = image_paths
    return label_to_imgs


def random_subset_label_dict(
    label_to_imgs, num_identities=10, max_imgs_per_identity=None
):
    all_identities = list(label_to_imgs.keys())
    random.shuffle(all_identities)

    chosen_identities = all_identities[:num_identities]

    subset_dict = {}
    for identity in chosen_identities:
        if identity in label_to_imgs:
            paths = label_to_imgs[identity]
            random.shuffle(paths)
            if max_imgs_per_identity is not None:
                paths = paths[:max_imgs_per_identity]
            subset_dict[identity] = paths

    return subset_dict


def compute_cosine_similarities(label_to_imgs, model, transform, device):
    """Compute cosine similarity scores for positive and negative pairs."""
    positive_similarities = []
    negative_similarities = []

    for label, img_paths in label_to_imgs.items():
        if len(img_paths) < 2:
            continue
        img1, img2 = random.sample(img_paths, 2)
        emb1 = get_embedding(img1, model, transform, device)
        emb2 = get_embedding(img2, model, transform, device)
        pos_sim = F.cosine_similarity(
            torch.tensor(emb1).unsqueeze(0), torch.tensor(emb2).unsqueeze(0)
        ).item()
        positive_similarities.append(pos_sim)

    all_labels = list(label_to_imgs.keys())
    num_negatives = len(positive_similarities)

    for _ in range(num_negatives):
        label1, label2 = random.sample(all_labels, 2)
        img1 = random.choice(label_to_imgs[label1])
        img2 = random.choice(label_to_imgs[label2])
        emb1 = get_embedding(img1, model, transform, device)
        emb2 = get_embedding(img2, model, transform, device)
        neg_sim = F.cosine_similarity(
            torch.tensor(emb1).unsqueeze(0), torch.tensor(emb2).unsqueeze(0)
        ).item()
        negative_similarities.append(neg_sim)

    similarities = np.concatenate(
        [np.array(positive_similarities), np.array(negative_similarities)]
    )
    true_labels = np.concatenate(
        [np.ones(len(positive_similarities)), np.zeros(len(negative_similarities))]
    )

    return similarities, true_labels


def find_best_threshold(similarities, true_labels):
    """Find the best threshold for classification."""
    thresholds = np.arange(0.0, 1.0, 0.01)
    best_thresh, best_acc = 0.0, 0.0
    acc_list = []

    for thresh in thresholds:
        preds = similarities > thresh
        acc = accuracy_score(true_labels, preds)
        acc_list.append(acc)
        if acc > best_acc:
            best_acc, best_thresh = acc, thresh

    return best_thresh, best_acc, thresholds, acc_list


def evaluate_model(model, device, dataset_path, test_transforms, name):
    """Evaluate the model and log results."""
    label_to_imgs = build_label_to_imgs(dataset_path)

    subset_dict = random_subset_label_dict(
        label_to_imgs, num_identities=10, max_imgs_per_identity=20
    )
    tsne_dataset = TSNEDataset(subset_dict, transform=test_transforms)
    tsne_dataloader = DataLoader(tsne_dataset, batch_size=32, shuffle=False)

    embeddings, labels = extract_embeddings(model, tsne_dataloader, device)
    plot_and_log_tsne(
        embeddings, labels, name=name, perplexity=40, artifact_name="tsne_" + name
    )

    similarities, true_labels = compute_cosine_similarities(
        label_to_imgs, model, test_transforms, device
    )

    best_thresh, best_acc, thresholds, acc_list = find_best_threshold(
        similarities, true_labels
    )

    pred_labels = similarities > best_thresh
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)

    fp = np.sum((pred_labels == 1) & (true_labels == 0))
    tn = np.sum((pred_labels == 0) & (true_labels == 0))
    fn = np.sum((pred_labels == 0) & (true_labels == 1))
    tp = np.sum((pred_labels == 1) & (true_labels == 1))
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    fpr_curve, tpr_curve, _ = roc_curve(true_labels, similarities, pos_label=1)
    roc_auc = auc(fpr_curve, tpr_curve)

    print(f"Optimal threshold: {best_thresh:.2f} with accuracy: {best_acc:.2f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"False Positive Rate: {fpr:.4f}")
    print(f"False Negative Rate: {fnr:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    wandb.log(
        {
            "best_threshold": best_thresh,
            "best_thresh_acc": best_acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "FPR": fpr,
            "FNR": fnr,
            "ROC_AUC": roc_auc,
        }
    )

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr_curve,
        tpr_curve,
        color="darkorange",
        lw=2,
        label=f"ROC curve (area = {roc_auc:.2f})",
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Face Verification (Cosine Similarity)")
    plt.legend(loc="lower right")

    os.makedirs("./ROCplots", exist_ok=True)
    plot_filename = f"./ROCplots/ROC_CURVE_{name}.png"
    plt.savefig(plot_filename)
    plt.close()

    artifact = wandb.Artifact("roc_plots", type="analysis")
    artifact.add_file(plot_filename)
    wandb.log_artifact(artifact)
