### **Face Authentication Pipeline**
A **deep learning-based face authentication pipeline** using **ResNet-50** for feature extraction and a **hybrid triplet loss** function for training. This project includes **data preprocessing, model training, evaluation, and deployment** via **FastAPI**.

---

## **📜 Table of Contents**
- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Training](#training)
- [API Usage](#api-usage)
- [Results](#results)
- [To-Do](#to-do)
- [Acknowledgments](#acknowledgments)

---

## **✨ Features**
✔ **Face Embedding Extraction** using a fine-tuned ResNet-50  
✔ **Triplet Loss with Hard Negative Mining** for robust training  
✔ **Dynamic Margin Scheduling** to improve training stability  
✔ **FastAPI for Real-Time Face Verification**  
✔ **MTCNN for Face Detection & Cropping**  
✔ **t-SNE Visualization of Embeddings**  
✔ **ROC AUC, F1 Score, Precision/Recall Metrics**  
✔ **Weights & Biases (W&B) for Experiment Tracking**  

---

## **⚡ Installation**
No Conda needed — plain **pip** or **uv** in a virtualenv. Requires **Python ≥ 3.11**.

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/mohamedkhayat/face-auth-pipeline.git
cd face-auth-pipeline
```

### **2️⃣ Create a virtualenv and install dependencies**
Dependencies in `requirements.txt` are pinned to a version set verified end-to-end.

#### **Option A: uv (recommended)**
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

#### **Option B: pip**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **GPU:** on Linux, the pinned `torch`/`torchvision` install **CUDA-enabled** wheels from PyPI by default. For a **CPU-only** machine, install the CPU builds first, then the rest (already-satisfied pins are skipped):
> ```bash
> pip install torch==2.14.1 torchvision==0.29.1 \
>     --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```
> For other platforms see the [PyTorch install guide](https://pytorch.org/get-started/locally/). Training automatically falls back to CPU when no GPU is available (e.g. `CUDA_VISIBLE_DEVICES="" python src/train.py`).

> **MTCNN:** the first API start downloads the MTCNN face-detector weights (~30 MB) automatically — you need internet access once.

---

## **🗂 Dataset**
Place face images in **one folder per identity** under `data/`:
```
data/
├── train/
│   ├── <identity_1>/  img1.jpg, img2.jpg, ...
│   ├── <identity_2>/  ...
└── val/
    ├── <identity_1>/  ...
    └── ...
```
Each identity must contain **at least 2 images**. To use a different location, set the `DATA_DIR` environment variable (defaults to `<repo>/data`).

---

## **🎯 Training**
```bash
python src/train.py
```
- Uses GPU if available, otherwise CPU.
- **W&B logging is offline by default**, so no account is needed. To log to the cloud: `wandb login` and `WANDB_MODE=online python src/train.py`.
- All hyperparameters and paths live in **`src/config.py`**. `EPOCHS`, `DATA_DIR`, `MODELS_DIR`, `MODEL_FILENAME`, and `WANDB_MODE` can be overridden with environment variables.
- The best model is saved to **`models/face_verification_resnet50.pth`**; t-SNE and ROC plots are written to `tsneplots/` and `ROCplots/`.

---

## **🚀 API Usage**
Train a model first (step above) — the API loads `models/face_verification_resnet50.pth` on startup. Then deploy with **FastAPI**:
```bash
uvicorn src.api:app --reload
```
Now, the API runs at `http://127.0.0.1:8000`

### **📌 Endpoints**
#### **1️⃣ Extract Face Embedding**
```http
POST /embedding
```
**Request:**
```bash
curl -X 'POST' 'http://127.0.0.1:8000/embedding' \
     -F 'file=@test_image.jpg'
```
**Response:**
```json
{
  "embedding": [0.1234, -0.5678, ...]
}
```

#### **2️⃣ Compare Two Faces**
```http
POST /compare
```
**Request:**
```bash
curl -X 'POST' 'http://127.0.0.1:8000/compare' \
     -F 'ref_embedding="[0.1234, -0.5678, ...]"' \
     -F 'test_file=@test_image2.jpg'
```
**Response:**
```json
{
  "similarity": 0.85,
  "result": "access_granted"
}
```
Decision thresholds are in `src/config.py`: `similarity ≥ THRESHOLD (0.9)` → `access_granted`, `≥ EMAIL_THRESHOLD (0.5)` → `email_verification`, otherwise `access_denied`.

---

## **🔬 Results**
| Metric           | Score |
|-----------------|-------|
| **Best Threshold** | `0.91` |
| **Accuracy**     | `89%` |
| **Precision**    | `86%` |
| **Recall**       | `92%` |
| **F1 Score**     | `89%` |
| **ROC AUC**      | `95%` |

![tsne_plotexperiment_11_03_18_56](https://github.com/user-attachments/assets/1ae679f5-3188-4509-9f26-ed85859f0792)

---

## **📌 To-Do**
✅ Improve the inference pipeline with liveness detection  
---

## **🙌 Acknowledgments**
-I gratefully acknowledge the use of the VGGFace2 dataset in this project. VGGFace2 is a large-scale face recognition dataset collected by the Visual Geometry Group at the University of Oxford, containing 3.31 million images of 9,131 subjects with large variations in pose, age, illumination, ethnicity, and profession. For more details, please refer to the original paper: [VGGFace2: A dataset for recognising faces across pose and age](https://www.robots.ox.ac.uk/~vgg/publications/2018/Cao18/cao18.pdf).
