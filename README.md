### **Face Authentication Pipeline**
A **deep learning-based face authentication pipeline** using **ResNet-50** for feature extraction and a **hybrid triplet loss** function for training. This project includes **data preprocessing, model training, evaluation, and deployment** via **FastAPI**.

---

## **📜 Table of Contents**
- [Features](#features)
- [Installation](#installation)
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
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/mohamedkhayat/face-auth-pipeline.git
cd face-auth-pipeline
```

### **2️⃣ Set Up the Environment**

#### **Option 1: Using `env.yml` (Recommended)**
If you prefer using Conda's environment file, run:
```bash
conda env create -f env.yml
conda activate facenetcorrect
```

#### **Option 2: Manual Installation**
Create a new Conda environment and install dependencies manually:
```bash
conda create --name facenet python=3.9 -y
conda activate facenet
pip install -r requirements.txt
```

> **⚠ Note**: If you plan to use a GPU, ensure you have **PyTorch installed with CUDA**. You can check PyTorch's official [installation guide](https://pytorch.org/get-started/locally/) for the correct setup.

---

## **🎯 Training**
1. Place your dataset in `./data/train/` and `./data/val/`
2. Run the training script:
```bash
python src/train.py
```
3. Training progress is logged in **Weights & Biases (W&B)**.

---

## **🚀 API Usage**
Once trained, deploy the model using **FastAPI**:
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


