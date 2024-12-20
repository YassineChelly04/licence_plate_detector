import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
import easyocr
import pandas as pd
from util import set_background, write_csv
import uuid
import os
from streamlit_webrtc import webrtc_streamer
import av

# Set background and load paths
set_background(os.path.join(BASE_DIR, "imgs", "background.png"))

folder_path = BASE_DIR
LICENSE_MODEL_DETECTION_DIR = os.path.join(BASE_DIR, "models", "mymodel11.pt")
COCO_MODEL_DIR = os.path.join(BASE_DIR, "models", "yolo11n.pt")


# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)



vehicles = [2]
header = st.container()
body = st.container()

# Load models
coco_model = YOLO(COCO_MODEL_DIR)
license_plate_detector = YOLO(LICENSE_MODEL_DETECTION_DIR)


# Initialize session state
if "state" not in st.session_state:
    st.session_state["state"] = "Uploader"

class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_to_an = img.copy()
        img_to_an = cv2.cvtColor(img_to_an, cv2.COLOR_RGB2BGR)
        license_detections = license_plate_detector(img_to_an)[0]

        if len(license_detections.boxes.cls.tolist()) != 0:
            for license_plate in license_detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate

                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

                license_plate_crop = img[int(y1):int(y2), int(x1): int(x2), :]
            
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY) 

                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_gray, img)

                cv2.rectangle(img, (int(x1) - 40, int(y1) - 40), (int(x2) + 40, int(y1)), (255, 255, 255), cv2.FILLED)
                cv2.putText(img,
                            str(license_plate_text),
                            (int((int(x1) + int(x2)) / 2) - 70, int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 0),
                            3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def read_license_plate(license_plate_crop, img):
    scores = 0
    detections = reader.readtext(license_plate_crop)

    width = img.shape[1]
    height = img.shape[0]
    
    if detections == []:
        return None, None

    rectangle_size = license_plate_crop.shape[0]*license_plate_crop.shape[1]

    plate = [] 

    for result in detections:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        
        if length*height / rectangle_size > 0.17:
            bbox, text, score = result
            text = result[1]
            text = text.upper()
            scores += score
            plate.append(text)
    
    if len(plate) != 0: 
        return " ".join(plate), scores/len(plate)
    else:
        return " ".join(plate), 0

def process_frame(img):
    license_numbers = 0
    results = {}
    licenses_texts = []
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    object_detections = coco_model(img)[0]
    license_detections = license_plate_detector(img)[0]

    # Car detection
    if len(object_detections.boxes.cls.tolist()) != 0:
        for detection in object_detections.boxes.data.tolist():
            xcar1, ycar1, xcar2, ycar2, car_score, class_id = detection

            if int(class_id) in vehicles:
                cv2.rectangle(img, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 0, 255), 3)
    else:
        xcar1, ycar1, xcar2, ycar2 = 0, 0, 0, 0
        car_score = 0

    # License plate detection
    if len(license_detections.boxes.cls.tolist()) != 0:
        license_plate_crops_total = []
        for license_plate in license_detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

            license_plate_crop = img[int(y1):int(y2), int(x1): int(x2), :]

            img_name = '{}.jpg'.format(uuid.uuid1())
         
            cv2.imwrite(os.path.join(folder_path, img_name), license_plate_crop)

            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY) 

            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_gray, img)

            licenses_texts.append(license_plate_text)

            if license_plate_text is not None and license_plate_text_score is not None:
                license_plate_crops_total.append(license_plate_crop)
                results[license_numbers] = {}
                
                results[license_numbers][license_numbers] = {
                    'car': {
                        'bbox': [xcar1, ycar1, xcar2, ycar2], 
                        'car_score': car_score
                    },
                    'license_plate': {
                        'bbox': [x1, y1, x2, y2],
                        'text': license_plate_text,
                        'bbox_score': score,
                        'text_score': license_plate_text_score
                    }
                } 
                license_numbers += 1
          
        write_csv(results, os.path.join(BASE_DIR, "csv_detections", "detections.csv"))

        img_wth_box = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        return [img_wth_box, licenses_texts, license_plate_crops_total]
    
    else: 
        img_wth_box = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return [img_wth_box]

def process_video(video_path):
    # Open the video
    cap = cv2.VideoCapture(video_path)
    
    # Create a list to store processed frames and results
    processed_frames = []
    all_licenses_texts = []
    all_license_plate_crops = []
    
    # Create a set to store unique license plate texts
    unique_license_plates = set()
    
    # Read frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to RGB (from BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = process_frame(frame_rgb)
        
        # Store results
        if len(results) == 3:
            processed_frames.append(results[0])
            
            # Check for unique license plates
            for license_plate_text in results[1]:
                if license_plate_text is not None and license_plate_text not in unique_license_plates:
                    unique_license_plates.add(license_plate_text)
                    all_licenses_texts.append(license_plate_text)
                    all_license_plate_crops.extend(results[2])
    
    cap.release()
    
    return processed_frames, all_licenses_texts, all_license_plate_crops

# State change functions
def change_state_uploader():
    st.session_state["state"] = "Uploader"
    
def change_state_camera():
    st.session_state["state"] = "Camera"
    
def change_state_live():
    st.session_state["state"] = "Live"

# UI Setup
with header:
    _, col1, _ = st.columns([0.2, 1, 0.1])
    col1.markdown("""
    <h1 style='text-align: center; font-size: 50px; white-space: nowrap; margin: 0;'>
        💥 License Car Plate Detection 🚗
    </h1>
    """, unsafe_allow_html=True)

    _, col4, _ = st.columns([0.1,1,0.2])
    col4.subheader("Computer Vision Detection with YOLOv11n 🧪")

    _, col, _ = st.columns([0.3,1,0.1])
    col.image(os.path.join(BASE_DIR, "imgs", "plate_test.jpg"))

    st.write("The different models detect the car and the license plate in a given image or video, then extract the info about the license using EasyOCR, and crop and save the license plate as an Image, with a CSV file with all the data.")

with body:
    _, col1, _ = st.columns([0.1,1,0.2])
    col1.subheader("Check Out the License Car Plate Detection Model 🔎!")

    _, colb1, colb2, colb3 = st.columns([0.2, 0.7, 0.6, 1])

    if colb1.button("Upload an Image/Video", on_click=change_state_uploader):
        pass
    elif colb2.button("Take a Photo", on_click=change_state_camera):
        pass
    elif colb3.button("Live Detection", on_click=change_state_live):
        pass

    # Input handling
    if st.session_state["state"] == "Uploader":
        img = st.file_uploader("Upload a Car Image or Video: ", type=["png", "jpg", "jpeg", "mp4", "avi"])
    elif st.session_state["state"] == "Camera":
        img = st.camera_input("Take a Photo: ")
    elif st.session_state["state"] == "Live":
        webrtc_streamer(key="sample", video_processor_factory=VideoProcessor)
        img = None

    _, col2, _ = st.columns([0.3,1,0.2])
    _, col5, _ = st.columns([0.8,1,0.2])

    # Processing
    if img is not None:
        # Check if it's a video
        if img.name.lower().endswith(('.mp4', '.avi', '.mov')):
            # Save the uploaded video temporarily
            temp_video_path = os.path.join(folder_path, f"temp_video_{uuid.uuid4()}.mp4")
            with open(temp_video_path, "wb") as f:
                f.write(img.getvalue())

            # Process video
            if col5.button("Apply Detection"):
                processed_frames, texts, license_plate_crops = process_video(temp_video_path)

                # Display results
                if processed_frames:
                    _, col3, _ = st.columns([0.4,1,0.2])
                    col3.header("Detection Results ✅:")

                    # Display first frame with detections
                    _, col4, _ = st.columns([0.1,1,0.1])
                    col4.image(processed_frames[0])

                    # Filter out None values
                    texts = [t for t in texts if t is not None]

                    if texts:
                        _, col9, _ = st.columns([0.4,1,0.2])
                        col9.header("License Plates Cropped ✅:")

                        _, col11, _ = st.columns([0.45,1,0.55])
                        
                        # Display license plates and their numbers
                        for i in range(min(len(texts), len(license_plate_crops))):
                            col9.image(license_plate_crops[i], width=350)
                            col11.success(f"License Number {i+1}: {texts[i]}")

                    # Read and display CSV
                    df = pd.read_csv(os.path.join(BASE_DIR, "csv_detections", "detections.csv"))
                    st.dataframe(df)

                    # Clean up temporary video file
                    os.remove(temp_video_path)

        # Image processing
        else:
            image = np.array(Image.open(img))    
            col2.image(image, width=400)

            if col5.button("Apply Detection"):
                results = process_frame(image)

                if len(results) == 3:
                    prediction, texts, license_plate_crop = results[0], results[1], results[2]

                    texts = [i for i in texts if i is not None]
                    
                    if len(texts) == 1 and len(license_plate_crop):
                        _, col3, _ = st.columns([0.4,1,0.2])
                        col3.header("Detection Results ✅:")

                        _, col4, _ = st.columns([0.1,1,0.1])
                        col4.image(prediction)

                        _, col9, _ = st.columns([0.4,1,0.2])
                        col9.header("License Cropped ✅:")

                        _, col10, _ = st.columns([0.3,1,0.1])
                        col10.image(license_plate_crop[0], width=350)

                        _, col11, _ = st.columns([0.45,1,0.55])
                        col11.success(f"License Number: {texts[0]}")

                        df = pd.read_csv(os.path.join(BASE_DIR, "csv_detections", "detections.csv"))
                        st.dataframe(df)
                    elif len(texts) > 1 and len(license_plate_crop) > 1:
                        _, col3, _ = st.columns([0.4,1,0.2])
                        col3.header("Detection Results ✅:")

                        _, col4, _ = st.columns([0.1,1,0.1])
                        col4.image(prediction)

                        _, col9, _ = st.columns([0.4,1,0.2])
                        col9.header("License Cropped ✅:")

                        _, col10, _ = st.columns([0.3,1,0.1])
                        _, col11, _ = st.columns([0.45,1,0.55])

                        for i in range(len(license_plate_crop)):
                            col10.image(license_plate_crop[i], width=350)
                            col11.success(f"License Number {i+1}: {texts[i]}")

                        df = pd.read_csv(os.path.join(BASE_DIR, "csv_detections", "detections.csv"))
                        st.dataframe(df)
