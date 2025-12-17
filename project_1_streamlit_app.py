# Tesla Autonomous Vision - YOLOv5 Object Detection
# Version: 1.0.1 - Streamlit 1.30.0 Compatible
# Author: Ratnesh Singh

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
from tesla_object_detection import YOLODetector, download_file, unzip_file

# --- Page Configuration ---
st.set_page_config(
    page_title="Tesla Vision | Object Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Aesthetics ---
st.markdown("""
<style>
    /* Global Settings */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }

    /* Premium Nature/Start-up Background */
    .stApp {
        background: linear-gradient(135deg, #0f1c15 0%, #0d2320 40%, #112d32 100%); /* Deep Forest/Green-Teal */
        background-attachment: fixed;
        color: #E0E0E0;
    }

    /* Tesla Solid Sidebar */
    [data-testid="stSidebar"] {
        background-color: #171A20; /* Official Tesla Dark Grey */
        border-right: 1px solid #333;
    }
    [data-testid="stSidebar"] h1, h2, h3 {
        color: #FFFFFF;
    }
    
    /* Headers & Typography */
    h1, h2, h3 {
        color: #FFFFFF;
        font-weight: 700;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    h1 {
        font-size: 2.5rem;
    }

    /* Primary Actions (Tesla Red Accent) */
    .stButton>button {
        background: cubic-bezier(0.25, 0.46, 0.45, 0.94);
        background-color: #E82127; 
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        box-shadow: 0 4px 15px rgba(232, 33, 39, 0.3); /* Red Glow */
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #ff2b32;
        box-shadow: 0 6px 20px rgba(232, 33, 39, 0.5);
        transform: translateY(-2px);
    }

    /* Modern Pills/Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        border-bottom: none;
        padding-bottom: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        height: auto;
        padding: 8px 16px;
        background-color: rgba(255, 255, 255, 0.05); /* Dark Glass */
        border-radius: 30px; /* Pill Shape */
        color: #cccccc;
        font-weight: 500;
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(255, 255, 255, 0.1);
        color: #ffffff;
        transform: translateY(-1px);
        border-color: rgba(255, 255, 255, 0.2);
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(232, 33, 39, 0.2); /* Red Tint */
        color: #E82127; /* Tesla Red Text */
        font-weight: 700;
        border: 1px solid #E82127; /* Red Border */
        box-shadow: 0 4px 10px rgba(232, 33, 39, 0.15);
    }

    /* Glass Cards for Metrics & Info */
    div[data-testid="metric-container"], .stInfo, .stSuccess {
        background-color: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stAlert {
         background-color: rgba(255, 255, 255, 0.03);
         border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Inputs */
    .stSlider > div > div > div > div {
        background-color: #E82127;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        background: #0f1c15;
    }
    ::-webkit-scrollbar-thumb {
        background: #333;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# --- Constants & Paths ---
MODEL_ZIP = 'YoloV5_ONNX.zip'
ONNX_MODEL = 'yolov5n.onnx' 
CLASSES_FILE = 'coco.names.txt'
VID_FILE = 'Vid_Self-Driving_Demo.mp4'

# IDs
VID_ID = '1KyOIRWMBE-oX7em0bhTPkBbbM5mAlKII'
MODEL_ID = '1g7BocB3bO9l3qk8pzBmn7F32A9ib3Tf2'

# --- Initialization & Resource Loading ---

@st.cache_resource
def load_detector():
    # Ensure files exist
    if not os.path.exists(ONNX_MODEL) or not os.path.exists(CLASSES_FILE):
        with st.spinner("Downloading Model Assets..."):
            download_file(MODEL_ID, MODEL_ZIP)
            unzip_file(MODEL_ZIP)
            
    if not os.path.exists(VID_FILE):
        with st.spinner("Downloading Demo Video..."):
            download_file(VID_ID, VID_FILE)
            
    # Load Model
    try:
        detector = YOLODetector(ONNX_MODEL, CLASSES_FILE)
        return detector
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

detector = load_detector()

# --- Sidebar Controls ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/e/e8/Tesla_logo.png", width=120)
st.sidebar.markdown("<br>", unsafe_allow_html=True)

# Table of Contents
st.sidebar.markdown("<h3 style='color: #E82127;'>📖 Table of Contents</h3>", unsafe_allow_html=True)
st.sidebar.markdown("""
*   **1. Project Scope & Objective**
*   **2. Single Stage Detectors (SSD)**
*   **3. YOLOv5 Architecture**
*   **4. Implementation Details**
*   **5. Performance & Results**
*   **6. Future Scope**
""")

st.sidebar.markdown("---")

st.sidebar.markdown("### 📑 Project Document")
pdf_path = "Object Detection SSM Tesla Driving Car.pdf"

if os.path.exists(pdf_path):
    with open(pdf_path, "rb") as f:
        pdf_data = f.read()
        st.sidebar.download_button(
            label="Download Report (PDF)",
            data=pdf_data,
            file_name="Tesla_Object_Detection_Report.pdf",
            mime="application/pdf",
            help="Click to download the full project documentation."
        )
else:
    st.sidebar.info("Project documentation file not found.")
temp_file = None

# --- Main UI ---
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("""
        <div style='display: flex; align-items: center; background-color: rgba(23, 26, 32, 0.95); padding: 15px 25px; border-radius: 12px; border-left: 5px solid #E82127; box-shadow: 0 4px 12px rgba(0,0,0,0.3); margin-bottom: 20px;'>
            <img src="https://upload.wikimedia.org/wikipedia/commons/e/e8/Tesla_logo.png" style="width: 40px; height: auto; margin-right: 20px; filter: brightness(0) invert(1);">
            <h1 style='color: #E82127; margin: 0; font-size: 2.2rem; padding: 0;'>Tesla Autonomous Car Driving</h1>
        </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
        <div style='background-color: rgba(23, 26, 32, 0.95); color: white; padding: 15px; border-radius: 12px; border-right: 5px solid #E82127; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.3); font-size: 0.85em; display: flex; align-items: center; justify-content: center; height: 100%; margin-bottom: 20px;'>
            <div>
                <strong style='color: #E82127; font-size: 1.1em;'>Ratnesh Singh</strong> <br> <span style='color: #E82127; font-size: 0.9em; font-weight: bold;'>Data Scientist (4+Year Exp)</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Create Tabs
tabs = st.tabs([
    "🚀 Project Demo", 
    "🎯 Objective", 
    "🧠 Introduction to SSD",
    "🧠 Introduction to YOLO",
    "💾 Model Details", 
    "🏁 Conclusion"
])

# --- Tab 1: Project Demo ---
with tabs[0]:
    st.markdown("<h3 style='color: #E82127;'>Object Detection Prototype (Single Stage Detector)</h3>", unsafe_allow_html=True)
    st.markdown("Real-time inference using **YOLOv5** on CPU to detect Vehicles, Pedestrians, and Traffic Signals.")

    # --- Control Panel ---
    st.markdown("<h4 style='color: #E82127;'>⚙️ Control Panel</h4>", unsafe_allow_html=True)
    
    # Config & Inputs
    c_input, c_settings = st.columns([1, 1])
    
    with c_input:
        st.caption("**Input Source**")
        input_source = st.radio("Select Source", ("Demo Video", "Upload Video"), label_visibility="collapsed")
        
        if input_source == "Upload Video":
            uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
            if uploaded_file is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file.read())
                video_path = tfile.name
                temp_file = tfile.name
        else:
             video_path = VID_FILE

    with c_settings:
        st.caption("**Model Settings**")
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.45, 0.05)
        nms_threshold = st.slider("NMS Threshold", 0.0, 1.0, 0.45, 0.05)
        
        st.markdown("<br>", unsafe_allow_html=True) # Spacer
        
        b1, b2 = st.columns(2)
        with b1:
             start_button = st.button("▶ Start Detection", type="primary", use_container_width=True)
        with b2:
             stop_button = st.button("⏹ Stop", use_container_width=True)

    st.markdown("---")

    # --- Live Feed Area ---
    col1, col2 = st.columns([3, 1])

    with col2:
        st.markdown("### Stats")
        status_indicator = st.empty()
        fps_metric = st.empty()
        status_indicator.info("Ready")
        fps_metric.metric("FPS", "0.00")

    with col1:
        st.markdown("### Live Feed")
        image_placeholder = st.empty()
        if not start_button:
            # Create a 640x360 black image (16:9 aspect ratio) to serve as a reliable placeholder
            default_img = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(default_img, "TESLA VISION SYSTEM", (160, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(default_img, "STANDBY", (260, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 1)
            image_placeholder.image(default_img, channels="BGR", use_container_width=True)

# --- Tab 2: Objective ---
with tabs[1]:
    st.header("Project Overview")
    
    st.markdown("""
    ### 🎯 Problem Statement
    You are working as a **Machine Learning Engineer in Tesla** as part of the Autonomous driving team. 
    The goal is to build a prototype ML model to identify and detect the location of various road objects in real-time.
    
    **Target Objects:**
    *   🚦 **Traffic Lights & Stop Signs**
    *   🚗 **Vehicles** (Cars, Bicycles, Trucks, etc.)
    *   🚶 **Pedestrians** (People)
    *   🐕 **Animals**
    """)
    
    st.markdown("---")
    
    st.markdown("### ⚡ Real-Time Constraints")
    st.info("""
    Our Object Detection Algorithm needs to be:
    1.  **Fast enough** to run inference in Real-Time on commodity hardware (**CPU only**).
    2.  **Highly accurate** to ensure safety.
    """)
    st.image("real_time_constraints.png", caption="Real-time Detection Example", use_container_width=True)
    

    st.markdown("---")
    st.header("Object Detection with Single Stage Detectors")
    st.markdown("""
    In the last lecture you learnt how Object detection with two stage detectors work.
    Despite being highly accurate, there are a few problems that Computer Visions practitioners found with Two Stage Detectors that limited it's real-time Application.
    """)

    st.subheader("Problems with RCNNs?")
    st.markdown("""
    1.  **Training required multiple phases**:
        *   First **Region Proposal Network (RPN)** needed to be trained in order to generate suggested bounding boxes.
        *   Then we train the actual classifier to recognize objects in images.
    2.  **Training took too long**:
        *   A (Faster) R-CNN consists of multiple components (Region Proposal Network, ROI Pooling Module, Final classifier).
        *   While all three fit together into a framework, they are still moving parts that slow down the entire training procedure.
    3.  **Inference time was too slow**:
        *   The final issue, and arguably most important, is that inference time was too slow — we could not yet obtain real-time object detection with deep learning.
    """)

    st.subheader("How can we fix these issues?")
    st.markdown("""
    This can be addressed by another family of object detection method known as **Single Stage Object Detection algorithms**.
    """)

    st.markdown("### What is Single Stage Object Detection?")
    st.info("""
    **Single-Stage Object Detectors** are a class of object detection architectures that are one-stage.
    They treat object detection as a simple **regression problem**.

    *   The input image fed to the network directly outputs the class probabilities and bounding box coordinates.
    *   These models **don't have the region proposal stage** (Region Proposal Network).
    """)
    st.image("C:/Users/rattu/Downloads/L8 P-2 ObjectDetection with Single Stage Methods/Project Run -1/single_stage_detection.png", caption="Single Stage Object Detection Architecture", use_container_width=True)

# --- Tab 3: Theory: SSD & YOLO ---
with tabs[2]:
    st.header("Algorithm Selection: Single Stage Detectors")
    
    st.markdown("""
    There are majorly two single Stage Detecion algorithms:
    1. **SSD**: Single Shot Detector
    2. **Yolo**: You Only Look Once

    Let's Dive Deeper into Each one of them:
    """)
    st.markdown("---")

    st.subheader("Introduction to SSD")
    st.markdown("""
    **Original paper**: [https://arxiv.org/pdf/1512.02325.pdf](https://arxiv.org/pdf/1512.02325.pdf)

    The **core idea behind the SSD network** is to have a **CNN** that takes in an image as input and produce detections at different **scales, shapes, and locations.**
    
    Let's understand this in detail:

    Before understanding how SSD works and detect object at diffferent scales, First let's observe the architecture:
    """)
    st.image("ssd_architecture.png", caption="SSD Architecture", use_container_width=True)

    st.markdown("""
    ### Architecture Details:
    * SSD’s architecture builds on primarily **VGG-16 architecture (Known as Base Network)**, but discards the fully connected layers.
    * We utilize the VGG layers up until conv_6 and then **detach all other layers, including the fully-connected layers**.
    * A set of **new CONV layers are then added to the architecture**—these are the layers that make the SSD framework possible.
    * As you can see from the diagram, each of these layers are **CONV layers as well** which means our Network is **Fully Convolutional**.

    Another thing to observe in the architecture is:
    1. We progressively **reduce the volume size in deeper layers**, as we would with a standard CNN.
    2. Each of the **CONV layer connects to the final detection layer**.

    The **fact** that **each feature map (Conv Layer) connects to the final detection layer** enabling the model to extract features at multiple scales and Size and progressively decrease the size of the input to each subsequent layer.
    """)
    
    st.markdown("---")
    
    st.subheader("Deep Dive: How SSD Detect Object at different Scales")
    st.image("ssd_scales.png", caption="Detections at Scales", use_container_width=True)

    st.markdown("""
    To produce detections of different locations in the image, **the SSD Network uses Grid Detectors**.
    * The `first two dimensions of the feature maps can be thought of as the grid size in which to divide the input image into`.
    * Hence, from above image you can see how last 2 Convoultion block of `5 * 5` and `3 * 3` divide the image into `5 * 5` and `3 * 3` **Grid** respectively.
    * This allows the **SSD network to detect objects at different locations for each feature maps layer**.

    Let's understand how B-Box prediction in SSD is done Next:
    """)

    st.markdown("### MultiBox Regression")
    st.markdown("""
    * The bounding box regression technique of SSD is inspired by Szegedy’s work on MultiBox [https://arxiv.org/abs/1412.1441], a method for fast class-agnostic bounding box coordinate proposals.
    * Interestingly, in the work done on MultiBox an Inception-style convolutional network is used.
    * The 1x1 convolutions that you see below help in dimensionality reduction since the number of dimensions go down (but “width” and “height” remains the same)
    """)
    st.image("multibox_loss.png", caption="MultiBox 1x1 Convolutions", use_container_width=True)

    st.markdown("""
    **MultiBox’s loss function combines two critical components that made their way into SSD:**
    
    1. **Confidence Loss**: this measures how confident the network is of the objectness of the computed bounding box. Categorical cross-entropy is used to compute this loss.
    2. **Location Loss**: this measures how far away the network’s predicted bounding boxes are from the ground truth ones from the training set. L2-Norm is used here.

    `multibox_loss = confidence_loss + alpha * location_loss`

    * To **produce detections of different shapes** in the image, **the SSD Network uses Default Boxes**: a set of predefined box of different scales and size:
    * There are `6 number of Default boxes` per feature map cell in the original paper, with its center placed at a certain offset from the grid cell (usually center).
    """)

    st.markdown("### Producing Detections")
    st.image("ssd_prediction_process.png", caption="SSD Prediction Process", use_container_width=True)

    st.markdown("""
    * You have seen how SSD takes input image and able to locate object **at different shapes, scales and locations**.
    * Now, all these feature maps are directly connected to **Detection Module.**
    * Each Feature Map is passed to **Detection module** and **Multiclass Classification + Bounding Box Regressor** is performed on each feature map to give results on image of `different shapes, scales and locations.`
    
    For Tensorflow Code Implementation of SSD, you can refer to:
    [https://github.com/balancap/SSD-Tensorflow](https://github.com/balancap/SSD-Tensorflow)
    """)

    st.markdown("### Beyond SSD:")
    st.markdown("""
    There has been a lot of research in Single Stage Detection methods.
    Next we are going to learn the most popular **Single Shot Detector** known as **YOLO**.
    """)

# --- Tab 4: Introduction to YOLO ---
# --- Tab 4: Introduction to YOLO ---
with tabs[3]:
    st.subheader("Introduction to YOLO algorithm")
    st.markdown("""
    As its name suggests, **YOLO (You Only Look Once)** applies a single forward pass neural network to the whole image and predicts the bounding boxes and their class probabilities as well.
    
    This technique makes YOLO quite fast without losing a lot of accuracies.
    """)

    st.subheader("Version history of Yolo:")
    st.markdown("""
    **In 2015**, Redmon J et al. Proposed the YOLO network, which is characterized by combining the candidate box generation and classification regression into a single step.
    *   **Original paper**: [https://arxiv.org/abs/1506.02640](https://arxiv.org/abs/1506.02640).
    *   Proposed architecture accelerated the speed of target detection, frame rate up to **45 fps**! 
    *   When predicting, the feature map is divided into **7x7 cells**, and each cell is predicted, which significantly reduces the calculation complexity.
    """)

    st.markdown("### Yolo V2:")
    st.markdown("""
    After a one year, Redmon J once proposed an improvised version: **YOLO9000 also known as YoloV2**.
    *   Compared to the previous generation, the mAP on the VOC2007 test set increased from 67.4% to 78.6%. 
    *   However, in yoloV2 as well a single cell is only responsible for predicting a single object facing the goal of overlap, the recognition was not good enough. [https://arxiv.org/abs/1612.08242](https://arxiv.org/abs/1612.08242)
    """)

    st.markdown("### Yolo V3:")
    st.markdown("""
    In **April 2018**, the author released the third version of YOLOv3:
    *   **Paper**: [https://arxiv.org/abs/1804.02767](https://arxiv.org/abs/1804.02767)
    *   The mAP-50 on the COCO dataset increased from 44.0% of YOLOv2 to 57.9%.
    *   Compared with **RetinaNet** (state of the art at the time with 61.1% mAP, ~98 ms/frame), YOLOv3 operates at **29 ms/frame** (input 416x416), resulting in a better speed-to-accuracy tradeoff.
    *   **Code**: [https://pjreddie.com/darknet/yolo/](https://pjreddie.com/darknet/yolo/)
    
    *Redmon stopped his research on Object-Detection and Yolo over concerns of his research being used for Military Purpose.*
    """)

    st.markdown("### Yolo V4:")
    st.markdown("""
    In **2020**, Bochkovskiy et al. took over YOLO Research and released YoloV4.
    *   Achieved state-of-the-art results: **43.5% mAP** on the MS COCO dataset at a real-time speed of ∼65 FPS on the Tesla V100 GPU.
    *   **Paper**: [https://arxiv.org/abs/2004.10934](https://arxiv.org/abs/2004.10934)
    *   **Code**: [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
    """)

    st.markdown("### Yolo V5:")
    st.markdown("""
    Within just two months of V4 Release, **Ultralytics** open sourced controversial YoloV5 without any official peer reviewed research paper.
    *   Developed using **PyTorch** framework unlike its predecessors which used Darknet framework.
    *   It soon gained popularity among applied community and is currently go-to framework for real-time Computer Vision Applications.
    *   **Code**: [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
    
    *There has been further advancements and version releases such as Scaled YoloV4, PP-Yolo, YoloX, YoloR and Recently launched YoloV6 and V7.*
    """)

    st.markdown("### Detailed Timeline:")
    st.image("yolo_timeline.png", caption="YOLO Version Timeline", use_container_width=True)
    
    st.markdown("---")

    st.subheader("Deep Dive into YOLO V3")
    st.info("We will deep dive into YOLO V3 as it serves as the Major Inspiration and baseline from which all version of Yolo has Evolved.")
    
    st.markdown("### The Idea behind YOLO v3:")
    st.markdown("""
    The author treats the object detection problem as a regression problem in the YOLO algorithm and divides the image into an **S × S grid**. 
    If the center of a target falls into a grid, the grid is responsible for detecting the target.
    """)
    st.image("yolo_dog_grid.png", caption="S x S Grid Logic", use_container_width=True)

    st.markdown("""
    **Each grid will output a bounding box, confidence, and class probability map.** Among them:
    
    1.  **Bounding Box**: Contains four values: x, y, w, h. (x, y) represents the center of the box. (W, h) defines the width and height.
    2.  **Confidence**: Indicates the probability of containing objects in this prediction box (IoU value between prediction and actual box).
    3.  **Class Probability**: Indicates the class probability of the object (YOLOv3 uses a two-class method).
    """)
    st.image("yolo_tesla_vector.png", caption="Output Vector Components", use_container_width=True)

    st.markdown("### Simple YOLO Architecture:")
    st.image("yolo_simple_arch.png", caption="Simple YOLO Architecture", use_container_width=True)

    st.markdown("### Yolo V3 Architecture details:")
    st.image("yolov3_full_arch.png", caption="YOLOv3 Architecture", use_container_width=True)

    st.markdown("""
    As mentioned in the original paper, YOLOv3 has **53 convolutional layers called Darknet-53** (shown below). 
    *   Mainly composed of Convolutional and Residual structures.
    *   Last three layers (Avgpool, Connected, Softmax) are used for ImageNet classification training and are **NOT used** when extracting features for detection.
    """)
    st.image("darknet53_table.png", caption="Darknet-53 Backbone", use_container_width=True)

    st.markdown("""
    **The reason behind picking Darknet-53 as the backbone:**
    1.  Comparable accuracy to advanced classifiers but with fewer floating-point operations and fastest calculation speed.
    2.  Speed is **1.5x of ResNet-101** and **2x of ResNet-152**.
    3.  Highest measurement floating-point operation per second (better GPU utilization).
    """)
    st.image("backbone_comparison.png", caption="Backbone Performance Comparison", use_container_width=True)

    st.markdown("### YOLO in easy steps:")
    st.markdown("""
    1.  **Divide the image into multiple grids** (e.g., 4x4 grids).
    2.  **Label the training data** as shown in the figure.
        *   If `C` is number of unique objects, `S*S` is number of grids, output vector length is `S*S*(C+5)`.
        *   Example: Target vector is 4x4x(3+5) for 3 classes (Car, Light, Pedestrian).
    3.  Make one deep convolutional neural net with **loss function as error between output activations and label vector**.
    4.  The model predicts the output of all the grids in just **one forward pass**.
    
    *   **Note**: The label for object being present in a grid cell is determined by the presence of object’s **centroid** in that grid.
    *   **Advantage**: Very fast prediction in a single pass.
    """)

    st.markdown("### Difference: SSD vs YOLO (Optional)")
    st.markdown("""
    *   Both detect images in a single pass.
    *   **YOLO**: Uses two fully connected layers.
    *   **SSD**: Uses multiple convolutional layers (feature layers at end of base network) to predict offsets for default boxes of different scales/aspect ratios.
    """)
    st.image("ssd_vs_yolo_comparison.png", caption="SSD vs YOLO", use_container_width=True)
    
    st.markdown("---")

    st.markdown("---")
    st.subheader("Let's go back to our problem statement and see how we can using Pre-Trained YoloV5 to solve it:")
    st.markdown("### Why YOLO V5:")
    st.markdown("""
    We are going to use pre-trained Yolo V5 models for our use case since they are trained using **COCO-Dataset** (https://cocodataset.org/#home) and already contain the Target-Classes which we want to Detect: - 'Car' , 'motorbike','aeroplane', 'bus', 'truck' - Traffic light - Person - so on..

    *   Due to it's choice of framework being **pytorch** it's highly popular in Applied CV Community.
    *   Pytorch models can be easily migrated to other frameworks such as onnx, tensorflow.
    *   It's faster than YoloV4 and has 5 checkpoints for different memory and speed requirements.

    For more details Check this blog: [https://blog.roboflow.com/yolov5-is-here/](https://blog.roboflow.com/yolov5-is-here/)
    """)

    st.markdown("### We will be using ONNX Format of YoloV5")
    st.markdown("#### What is ONNX?")
    st.info("""
    **ONNX (Open Neural Network Exchange)** is an open format built to represent machine learning models. 
    ONNX defines a common set of operators - the building blocks of machine learning and deep learning models - and a common file format to enable AI developers to use models with a variety of frameworks, tools, runtimes, and compilers.

    **Benefit:** Using ONNX, solution developed using one framework can be easily deployed in another framework and vice versa.
    """)

# --- Tab 5: Model Details ---
with tabs[4]:
    st.header("Implementation Details")
    
    st.markdown("### Why YOLOv5?")
    st.markdown("""
    We selected **YOLOv5** for this prototype for several key reasons:
    1.  **Pre-Trained on COCO**: It comes pre-trained on the COCO dataset, effectively recognizing our target classes (Car, Truck, Person, Traffic Light).
    2.  **PyTorch & ONNX**: Highly portable. We converted the model to **ONNX** format for universal deployment.
    3.  **Efficiency**: The **Nano (n)** model weight is extremely lightweight (~4MB), allowing strict Real-Time performance on CPU.
    """)

    st.markdown("### What is ONNX?")
    st.info("""
    **ONNX (Open Neural Network Exchange)** is an open format built to represent machine learning models. 
    
    *   ONNX defines a common set of operators - the building blocks of machine learning and deep learning models.
    *   It enables AI developers to use models with a variety of frameworks, tools, runtimes, and compilers.
    *   **Benefit**: A solution developed using one framework (like PyTorch) can be easily deployed in another framework (like OpenCV DNN) using ONNX.
    
    [Learn more at onnx.ai](https://onnx.ai/)
    """)
    
    st.markdown("### Deployment Pipeline")
    st.code("""
    # Forward Pass Logic
    1. Resize Input -> 640x640 (Blob)
    2. Model Inference -> Prediction Vectors
    3. Post-Process:
       - Filter by Confidence Threshold (>0.45)
       - Apply Non-Maximum Suppression (NMS) to remove duplicates
    4. Annotate & Display
    """, language="python")
    
    st.markdown("### Dataset")
    st.markdown("We validated the model using the **Self-Driving Car Dataset**, comprising crowd-sourced driving footage.")

    st.subheader("Let's plot top two images in dataset")
    st.image("dataset_samples.png", caption="Sample Images from Dataset", use_container_width=True)

    st.subheader("Let's setup config for model")
    st.markdown("""
    It did a great job of predicting **Vehicles and Traffic light** with just the pre-trained model and
    Every image was processed in less than a Second on CPU without any specialized hardware (1000 ms = 1 second).
    
    To Enhance performance we can finetune the model on our custom dataset as well.
    """)
    st.image("result_sample_1.png", caption="Prediction 1", use_container_width=True)
    st.image("result_sample_2.png", caption="Prediction 2", use_container_width=True)
    st.image("result_sample_3.png", caption="Prediction 3", use_container_width=True)
    

# --- Tab 6: Conclusion ---
with tabs[5]:
    st.header("Conclusion & Future Work")

    st.markdown("### Video Inference: (Optional)")
    st.markdown("""
    Let's Run our Inference on Video: Note that Every Second of a Video Contains 24+ Frames i.e. 24 Images so it might take around 20 mins to process the whole video on CPU.

    We can sample a small number of frames from every second depending on the speed of Change in Business-case at hand to make video inference Even Faster.

    [Demo on working video](https://drive.google.com/file/d/1QAMcZSNjJDqVTnFEDR0RRFFa5COJ5ay-/view?usp=sharing)
    """)

    st.markdown("### Disadvantage of Single Stage Models and need for retinanet:")
    st.markdown("""
    **Class imbalance problem**
    *   Both one stage detection methods, like SSD and YOLO evaluate almost 10^4 to 10^5 candidate locations per image.
    *   But only a few locations contain objects (i.e. Foreground) and rest are just background objects.
    *   This leads to **class imbalance problem**.
    *   And this turn out to be the central cause of making performance of one stage detectors inferior

    **Small objects and close-by objects may be missed by YOLO like algorithms**

    Hence , researchers have introduced **RetinaNet Model**. To learn more about Retina Net, refer to the post read Notebook: [RetinaNet Notebook](https://drive.google.com/file/d/1cg42_iwvzeyC-3OPcvel4-i0F00_4snG/view?usp=sharing)
    """)

# --- Logic Implementation for Demo ---
if start_button:
    # Need to target the placeholders defined in Tab 1
    if detector:
        status_indicator.markdown("🟢 **Status:** Running")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            st.error(f"Error opening video file: {video_path}")
        else:
            prev_time = time.time()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    status_indicator.markdown("⚪ **Status:** Finished")
                    break
                
                # Perform Detection
                output_frame = detector.detect(frame.copy())
                
                # FPS Calculation
                curr_time = time.time()
                try:
                    fps = 1 / (curr_time - prev_time)
                except ZeroDivisionError:
                    fps = 0
                prev_time = curr_time
                
                # Update Stats
                fps_metric.metric("FPS", f"{fps:.2f}")
                
                # Convert BGR to RGB for Streamlit
                output_frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
                
                # Display
                image_placeholder.image(output_frame_rgb, channels="RGB", use_container_width=True)
                
            cap.release()
            if temp_file:
                os.remove(temp_file)
    else:
        st.error("Detector not initialized.")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #E82127; padding: 30px;'>
    <p style="font-size: 1.2rem; font-weight: bold;">🚀 Deep Learning of Computer Vision - Object Detection - YOLO Algo</p>
    <p style='font-size: 0.9rem; margin-top: 15px; font-weight: bold; color: #FFFFFF; background-color: rgba(255, 255, 255, 0.1); padding: 8px 20px; border-radius: 20px; display: inline-block;'>
        Built with ❤️ by Ratnesh Singh | Data Scientist (4+Year Experience)
    </p>
    <p style="font-size: 1.0rem; margin-top: 10px; font-weight: bold;">
        Technologies Used: Python, Streamlit, PyTorch, ONNX, OpenCV, NumPy
    </p>
    <p style="font-size: 1.0rem; margin-top: 10px; font-weight: bold;">
        Project Workflow: Problem Definition &rarr; Model Selection (YOLOv5) &rarr; Data Validation &rarr; Implementation (ONNX) &rarr; Deployment & Testing
    </p>
</div>
""", unsafe_allow_html=True)
