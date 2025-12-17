import cv2
import numpy as np
import os
import time
import sys
import subprocess

# Constants
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.45

# Text parameters
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
THICKNESS = 1
BOX_COLOR = (0, 255, 255) # Yellow
FONT_COLOR = (0, 0, 0) # Black

def download_file(file_id, output_path):
    if os.path.exists(output_path):
        print(f"File {output_path} already exists. Skipping download.")
        return
    
    print(f"Downloading {output_path}...")
    try:
        subprocess.check_call(["gdown", file_id, "-O", output_path])
        print("Download complete.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading file: {e}")
        sys.exit(1)

def unzip_file(zip_path, extract_to="."):
    if not os.path.exists(zip_path):
        print(f"Zip file {zip_path} not found.")
        return
    
    print(f"Unzipping {zip_path}...")
    try:
        # Check if unzip is available, otherwise use python zipfile
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Unzip complete.")
    except Exception as e:
        print(f"Error unzipping file: {e}")

class YOLODetector:
    def __init__(self, model_path, classes_path, input_width=640, input_height=640):
        self.input_width = input_width
        self.input_height = input_height
        self.classes = self.load_classes(classes_path)
        self.net = cv2.dnn.readNet(model_path)
        
        # Enable CUDA if available (though problem statement says CPU only, it's good practice)
        # self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        # self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        
        # Force CPU as per requirements
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def load_classes(self, path):
        with open(path, 'rt') as f:
            return f.read().rstrip('\n').split('\n')

    def detect(self, image):
        blob = cv2.dnn.blobFromImage(image, 1/255, (self.input_width, self.input_height), [0, 0, 0], 1, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        return self.post_process(image, outputs)

    def post_process(self, input_image, outputs):
        class_ids = []
        confidences = []
        boxes = []

        rows = outputs[0].shape[1]
        image_height, image_width = input_image.shape[:2]

        x_factor = image_width / self.input_width
        y_factor = image_height / self.input_height

        # Iterate through detections.
        # Note: YOLOv5 ONNX output shape is usually (1, 25200, 85) for 80 classes, or generally (1, rows, classes+5)
        # The outputs[0] is (1, 25200, 85). We act on outputs[0][0] which is (25200, 85)
        
        predictions = outputs[0][0]
        
        for r in range(rows):
            row = predictions[r]
            confidence = row[4]

            if confidence >= SCORE_THRESHOLD:
                classes_scores = row[5:]
                class_id = np.argmax(classes_scores)

                if (classes_scores[class_id] > CONFIDENCE_THRESHOLD):
                    confidences.append(confidence)
                    class_ids.append(class_id)
                    cx, cy, w, h = row[0], row[1], row[2], row[3]

                    left = int((cx - w/2) * x_factor)
                    top = int((cy - h/2) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)

                    boxes.append([left, top, width, height])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

        for i in indices:
            # Handle different opencv versions where indices might be a list of lists or just a list
            idx = i if isinstance(i, (int, np.integer)) else i[0]
            
            box = boxes[idx]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            
            self.draw_prediction(input_image, class_ids[idx], confidences[idx], left, top, width, height)

        return input_image

    def draw_prediction(self, img, class_id, confidence, x, y, w, h):
        label = f"{self.classes[class_id]}:{confidence:.2f}"
        cv2.rectangle(img, (x, y), (x + w, y + h), BOX_COLOR, 2)
        
        text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
        dim, baseline = text_size[0], text_size[1]
        
        cv2.rectangle(img, (x, y), (x + dim[0], y - dim[1] - baseline), BOX_COLOR, cv2.FILLED)
        cv2.putText(img, label, (x, y - baseline), FONT_FACE, FONT_SCALE, FONT_COLOR, THICKNESS, cv2.LINE_AA)

def main():
    print("Tesla Object Detection Prototype - Loading...")
    
    # Files
    VID_FILE = 'Vid_Self-Driving_Demo.mp4'
    MODEL_ZIP = 'YoloV5_ONNX.zip'
    ONNX_MODEL = 'yolov5n.onnx' # Using Nano for CPU real-time
    CLASSES_FILE = 'coco.names.txt'
    
    # IDs
    VID_ID = '1KyOIRWMBE-oX7em0bhTPkBbbM5mAlKII'
    MODEL_ID = '1g7BocB3bO9l3qk8pzBmn7F32A9ib3Tf2'
    
    # Download Assets
    download_file(VID_ID, VID_FILE)
    if not os.path.exists(ONNX_MODEL) or not os.path.exists(CLASSES_FILE):
        download_file(MODEL_ID, MODEL_ZIP)
        unzip_file(MODEL_ZIP)

    if not os.path.exists(CLASSES_FILE):
         # Sometimes it extracts to a folder? Check root.
         # The zip usually contains yolov5*.onnx and coco.names.txt
         # Renaming logic if necessary
         pass

    # Initialize Detector
    print("Initializing YOLOv5 Nano Model...")
    try:
        detector = YOLODetector(ONNX_MODEL, CLASSES_FILE)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Video Capture
    cap = cv2.VideoCapture(VID_FILE)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    # Output Video Writer
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    out = cv2.VideoWriter('tesla_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    print(f"Processing video: {VID_FILE}")
    print("Press 'q' to stop.")
    
    start_time = time.time()
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_start = time.time()
        
        # Inference
        output_frame = detector.detect(frame)
        
        # FPS Calculation
        frame_end = time.time()
        proc_time = frame_end - frame_start
        fps_curr = 1.0 / proc_time
        
        # Draw FPS and Status
        cv2.putText(output_frame, f"Tesla Vision | FPS: {fps_curr:.2f}", (20, 40), FONT_FACE, 0.7, (0, 0, 255), 2)
        cv2.putText(output_frame, "Mode: Object Detection (CPU)", (20, 70), FONT_FACE, 0.6, (0, 255, 0), 1)

        # Write and Display
        out.write(output_frame)
        
        # Resize for display if too large
        display_frame = cv2.resize(output_frame, (1280, 720))
        cv2.imshow('Tesla Object Detection', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames. Current FPS: {fps_curr:.2f}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    total_time = time.time() - start_time
    print(f"Processing complete. Total frames: {frame_count}, Time: {total_time:.2f}s, Avg FPS: {frame_count/total_time:.2f}")
    print("Output saved to 'tesla_output.mp4'")

if __name__ == "__main__":
    main()
