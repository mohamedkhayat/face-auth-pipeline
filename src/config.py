"""Central configuration: all paths and hyperparameters live here.

Paths are resolved relative to the project root (not the CWD), so the
scripts work no matter where they are launched from. The main ones can
be overridden with environment variables (DATA_DIR, MODELS_DIR,
MODEL_FILENAME, WANDB_MODE, EPOCHS).
"""

import os
from typing import Literal

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(PROJECT_ROOT, "data"))
TRAIN_DATA_DIR = os.path.join(DATA_DIR, "train")
VAL_DATA_DIR = os.path.join(DATA_DIR, "val")

MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(PROJECT_ROOT, "models"))
TSNE_PLOTS_DIR = os.path.join(PROJECT_ROOT, "tsneplots")
ROC_PLOTS_DIR = os.path.join(PROJECT_ROOT, "ROCplots")

# File name train.py saves the best model as and the API loads at startup.
MODEL_FILENAME = os.environ.get("MODEL_FILENAME", "face_verification_resnet50.pth")
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILENAME)

# ---------------------------------------------------------------------------
# Weights & Biases — offline by default so training works out of the box.
# Run `wandb login` and `WANDB_MODE=online` to log to the cloud.
# ---------------------------------------------------------------------------
def _wandb_mode() -> Literal["online", "offline", "disabled", "shared"]:
    mode = os.environ.get("WANDB_MODE", "offline")
    if mode in ("online", "disabled", "shared"):
        return mode
    return "offline"


WANDB_PROJECT = "face-verification"
WANDB_MODE = _wandb_mode()

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 64 * 3
LR = 2e-4
LAYER4_LR = 5e-5  # learning rate for the unfrozen backbone.layer4
DECAY = 1e-3
DROPOUT = 0.3
EPOCHS = int(os.environ.get("EPOCHS", "65"))
FACTOR = 0.75  # ReduceLROnPlateau factor
EMB_DIM = 512

# API decision thresholds on cosine similarity
THRESHOLD = 0.9  # sim >= THRESHOLD -> access_granted
EMAIL_THRESHOLD = 0.5  # sim >= EMAIL_THRESHOLD -> email_verification

# Hybrid triplet loss
ALPHA = 0.4  # weight of the hard-triplet part
MARGIN = 0.2
INITIAL_MARGIN = 1.0
PEAK_MARGIN = 0.6  # maybe make this a bit higher
FINAL_MARGIN = 0.3  # maybe lower this
N_EPOCHS_MARGIN = 49  # epochs the margin schedule runs over
MARGIN_PEAK_EPOCH = 15
COSINE_START_EPOCH = 10  # cosine-similarity loss is added after this epoch
COSINE_WEIGHT = 0.25

# Early stopping
EARLY_STOP_PATIENCE = 5
EARLY_STOP_MIN_DELTA = 0.001
EARLY_STOP_WARMUP = 10  # early stopping is only checked from this epoch on

# Dataset
TRAIN_IDENTITY_FRACTION = 1.0  # fraction of training identities to use
SAMPLES_PER_IDENTITY = 10  # triplets drawn per identity per epoch

# Inference / evaluation
INPUT_SIZE = 224  # model input resolution
TRAIN_RESIZE = 256  # long-edge resize before the random training crop
TRAIN_ROTATION_DEG: float = 10  # max degrees for RandomRotation augmentation
TRAIN_AFFINE_DEG: float = 5  # max degrees for RandomAffine augmentation
CROP_MARGIN_PCT = 0.2  # MTCNN crop expansion as a fraction of box size
EVAL_BATCH_SIZE = 32
EVAL_IDENTITY_SUBSET = 10  # identities sampled for eval / t-SNE
EVAL_MAX_IMGS_PER_IDENTITY = 20
TSNE_PERPLEXITY = 40

# Dataloaders
NUM_WORKERS = min(4, (os.cpu_count() or 1) // 2)
