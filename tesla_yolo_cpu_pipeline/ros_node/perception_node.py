# ROS-style perception node (pseudo)
# Subscribe: /camera/image_raw
# Publish: /perception/objects

import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def callback(image):
    results = model(image)
    print(results)
