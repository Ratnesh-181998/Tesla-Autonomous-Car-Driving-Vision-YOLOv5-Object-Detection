import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm.auto import tqdm
import os
import sys

# Redirect stdout to output.txt
sys.stdout = open('output.txt', 'w')

print("Starting Object Detection Script...")
print("Note: Using Pre-trained YoloV5 models (yolov5n.onnx, yolov5s.onnx). No training is required.")

# Define content paths
CONTENT_DIR = "content"
IMAGES_DIR = os.path.join(CONTENT_DIR, "selfdriving cars data/images")
COCO_NAMES = os.path.join(CONTENT_DIR, "coco.names.txt")
YOLO_NANO = os.path.join(CONTENT_DIR, "yolov5n.onnx")
YOLO_SMALL = os.path.join(CONTENT_DIR, "yolov5s.onnx")
VIDEO_PATH = os.path.join(CONTENT_DIR, "Vid_Self-Driving_Demo.mp4")

# Check if directories exist
if not os.path.exists(IMAGES_DIR):
    print(f"Warning: Images directory not found at {IMAGES_DIR}")
if not os.path.exists(COCO_NAMES):
    print(f"Warning: coco.names.txt not found at {COCO_NAMES}")

print("Sample Images: ")
sample_imgs = glob(os.path.join(IMAGES_DIR, '*jpg'))[:5]
print(sample_imgs)

# Plot top two images and save them
print("Saving sample dataset images to 'sample_dataset_images.png'...")
for i, image_path in enumerate(glob(os.path.join(IMAGES_DIR, '*jpg'))[:2]):
    plt.figure(figsize=(12,8))
    img = plt.imread(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Sample Image {i+1}")
    plt.savefig(f"sample_dataset_image_{i+1}.png")
    plt.close()

# Load class names
classes = None
if os.path.exists(COCO_NAMES):
    with open(COCO_NAMES, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
    print("Class mappings loaded:")
    print(dict(enumerate(classes)))
else:
    print("Classes file not found.")

# Configuration
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
OBJECT_SCORE_THRESHOLD = 0.5
CLASS_CONFIDENCE_THRESHOLD = 0.45
NMS_THRESHOLD = 0.45
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.4
THICKNESS = 1
BOX_COLOR = (0,255,255)
FONT_COLOR= (0,0,0)

def draw_label(im, label, x, y):
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    cv2.rectangle(im, (x,y), (x + dim[0], y - (dim[1]+baseline) ), (255,255,255), cv2.FILLED);
    cv2.putText(im, label, (x, y - dim[1]+baseline ), FONT_FACE, FONT_SCALE, FONT_COLOR, THICKNESS, cv2.LINE_AA)

def yolo_forward_pass(input_image, net):
    blob = cv2.dnn.blobFromImage(input_image, 1/255,  (INPUT_WIDTH, INPUT_HEIGHT), [0,0,0], 1, crop=False)
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())
    return outputs

def post_process_outputs(input_image, outputs):
    class_ids = []
    confidences = []
    boxes = []
    rows = outputs[0].shape[1]
    image_height, image_width = input_image.shape[:2]
    x_factor = image_width / INPUT_WIDTH
    y_factor =  image_height / INPUT_HEIGHT

    for r in range(rows):
        row = outputs[0][0][r]
        confidence = row[4]
        if confidence >= OBJECT_SCORE_THRESHOLD:
            classes_scores = row[5:]
            class_id = np.argmax(classes_scores)
            if (classes_scores[class_id] > CLASS_CONFIDENCE_THRESHOLD):
                confidences.append(confidence)
                class_ids.append(class_id)
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                left = int((cx - w/2) * x_factor)
                top = int((cy - h/2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CLASS_CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        cv2.rectangle(input_image, (left, top), (left + width, top + height), BOX_COLOR, 2*THICKNESS)
        if classes:
             label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])
             draw_label(input_image, label, left, top)

    return input_image

# Process specific sample images
if os.path.exists(IMAGES_DIR):
    # Just take first 6 images from the directory since hardcoded paths might be different
    sample_images = glob(os.path.join(IMAGES_DIR, '*jpg'))[:6]
    
    if os.path.exists(YOLO_NANO):
        net = cv2.dnn.readNet(YOLO_NANO)
        
        print("Running inference on sample images...")
        for i, image_path in enumerate(sample_images):
            frame = cv2.imread(image_path)
            if frame is None: continue
            
            detections = yolo_forward_pass(frame, net)
            pred_img = post_process_outputs(frame.copy(), detections)
            
            t, _ = net.getPerfProfile()
            label = 'Inference time: %.2f ms' % (t * 1000.0 /  cv2.getTickFrequency())
            cv2.putText(pred_img, label, (10, 20), FONT_FACE, FONT_SCALE,  (255, 0, 50), THICKNESS, cv2.LINE_AA)
            
            # Save the inference result instead of showing
            fig = plt.figure(figsize=(14,8))
            ax1 = fig.add_subplot(1,2,1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB )
            ax1.imshow(frame_rgb)
            ax1.axis('off')
            ax1.set_title("Original")
            
            ax2 = fig.add_subplot(1,2,2)
            pred_img_rgb = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB )
            ax2.imshow(pred_img_rgb)
            ax2.axis('off')
            ax2.set_title("Detection")
            
            save_path = f"detection_result_{i+1}.png"
            plt.savefig(save_path)
            plt.close(fig)
            print(f"Saved detection result to {save_path}")

# Video Inference
def count_frames(video_path, manual=False):
    stream = cv2.VideoCapture(video_path)
    def manual_count(handler):
        frames = 0
        while True:
            status, frame = handler.read()
            if not status: break
            frames += 1
        return frames

    if manual:
        frames = manual_count(stream)
    else:
        try:
            frames = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
        except:
            frames = manual_count(stream)
    stream.release()
    return frames

if os.path.exists(VIDEO_PATH) and os.path.exists(YOLO_SMALL):
    print(f"Processing video: {VIDEO_PATH}")
    frame_count = count_frames(VIDEO_PATH, manual=True)
    print("Total Frames: ", frame_count)

    stream = cv2.VideoCapture(VIDEO_PATH)
    ret, img_src = stream.read()
    if ret:
        output_path = 'output.mp4'
        print(f"Video writer initialized for {output_path}")
        output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (img_src.shape[1],img_src.shape[0]))
        
        net = cv2.dnn.readNet(YOLO_SMALL)
        
        # tqdm output goes to stderr, so it won't clutter output.txt
        with tqdm(total=frame_count, file=sys.stderr) as pbar:
            stream.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset
            while True:
                ret, image_np = stream.read()
                if not ret: break
                
                detections = yolo_forward_pass(image_np, net)
                img = post_process_outputs(image_np.copy(), detections)
                
                t, _ = net.getPerfProfile()
                label = 'Inference time: %.2f ms' % (t * 1000.0 /  cv2.getTickFrequency())
                cv2.putText(img, label, (10, 20), FONT_FACE, FONT_SCALE,  (0, 255, 200), THICKNESS, cv2.LINE_AA)
                
                output.write(img)
                pbar.update(1)
                
        stream.release()
        output.release()
        print(f"Processing Complete. Saved to {output_path}")
    else:
        print("Failed to read video")
else:
    print("Video or Model file not found for video inference.")
