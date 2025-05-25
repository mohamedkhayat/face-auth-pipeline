import io
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import uvicorn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from torchvision.models import resnet50
from facenet_pytorch import MTCNN
import json
import src.config as config
from src.model import FaceVerificationModel


def load_and_preprocess_image(image_bytes: bytes, transform):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot open image : {str(e)}")

    box, _ = mtcnn.detect(img)

    if box is not None and len(box) > 0:
        x1, y1, x2, y2 = map(int, box[0])
        margin_pct = 0.2
        box_w = x2 - x1
        box_h = y2 - y1
        x1 = max(int(x1 - margin_pct * box_w), 0)
        y1 = max(int(y1 - margin_pct * box_h), 0)
        x2 = min(int(x2 + margin_pct * box_w), img.width)
        y2 = min(int(y2 + margin_pct * box_h), img.height)
        cropped_img = img.crop((x1, y1, x2, y2))

    else:
        cropped_img = img

    cropped_img = cropped_img.resize((224, 224))
    processed_img = transform(cropped_img)

    return processed_img.unsqueeze(0)


transforms = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

backbone = resnet50()
model = FaceVerificationModel(backbone, dropout=0.3, embedding_size=config.EMB_DIM).to(
    device
)

mtcnn = MTCNN(keep_all=False, device=device)

model.load_state_dict(
    torch.load("./models/face_verification_resnet50.pth", map_location=device)
)
model = model.to(device)
model.eval()

app = FastAPI(title="Face Verification API")


@app.post("/embedding")
async def get_embedding(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        processed_img = load_and_preprocess_image(image_bytes, transforms).to(device)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    with torch.no_grad():
        embedding = model(processed_img)

    embedding_list = embedding.squeeze(0).cpu().tolist()

    return JSONResponse(content={"embedding": embedding_list})


@app.post("/compare")
async def compare_faces(
    ref_embedding: str = Form(...), test_file: UploadFile = File(...)
):
    try:
        test_bytes = await test_file.read()
        test_img = load_and_preprocess_image(test_bytes, transforms).to(device)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        """ref_embedding_bytes = base64.b64decode(ref_embedding_str.encode("utf-8"))
        ref_emb = pickle.loads(ref_embedding_bytes)
        ref_emb = torch.tensor(ref_emb,device=device)"""
        ref_embedding_list = json.loads(ref_embedding)
        ref_emb = torch.tensor(ref_embedding_list, device=device)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error reading embedding : {str(e)}"
        )

    with torch.no_grad():
        test_emb = model(test_img)

    sim = F.cosine_similarity(ref_emb, test_emb).item()

    high_threshold = 0.9
    low_threshold = 0.5

    if sim > high_threshold:
        access = "access_granted"
    elif sim >= low_threshold:
        access = "email_verification"
    else:
        access = "access_denied"
    print(access, sim)
    return JSONResponse(content={"similarity": sim, "result": access})
