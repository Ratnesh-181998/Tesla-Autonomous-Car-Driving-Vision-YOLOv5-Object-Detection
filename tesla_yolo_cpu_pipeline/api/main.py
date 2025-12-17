from fastapi import FastAPI, UploadFile
import cv2, numpy as np
from ultralytics import YOLO

app = FastAPI()
model = YOLO("yolov8n.pt")

@app.post("/detect")
async def detect(file: UploadFile):
    img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), 1)
    results = model(img)
    return results[0].boxes.data.tolist()
