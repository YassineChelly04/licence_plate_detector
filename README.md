---

### **README.md**

# 🚗 Car Plate Detection System  
This project is a **Car Plate Detection System** designed for detecting **Tunisian license plates** using **YOLOv8** for object detection and **EasyOCR** for text recognition. It is built using Streamlit to create an interactive web application with live video stream support.

**Note**: While the system provides good accuracy in detecting license plates, the text extraction from images (license plate numbers) may not always be accurate due to limitations with EasyOCR.

---

## 📋 Features  
- **Real-Time Car Plate Detection**: Uses YOLOv8 for accurate and fast object detection.  
- **Text Recognition**: Reads and extracts the text (license plate) from the detected regions using EasyOCR.  
- **CSV Logging**: Saves detected license plates with timestamps in a CSV file.  
- **Web Interface**: Interactive interface powered by Streamlit.  
- **Live Streaming**: Processes live video streams using Streamlit-WebRTC.

---

## 🛠️ Technologies Used  
- **Python**  
- **Streamlit** (Web Interface)  
- **Ultralytics YOLOv8** (Object Detection)  
- **EasyOCR** (Text Recognition)  
- **OpenCV** (Image/Video Processing)  
- **Pandas** (CSV File Management)  
- **Streamlit-WebRTC** (Real-Time Video Streaming)

---

## 🚀 Installation  

### Prerequisites  
Ensure Python 3.8+ is installed on your machine.

### Steps  

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/your-username/car-plate-detection.git
   cd car-plate-detection
   ```

2. **Install Dependencies**  
   Run the following command to install required libraries:  
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**  
   Start the Streamlit app:  
   ```bash
   streamlit run app.py
   ```

---

## 📂 File Structure  
```
car-plate-detection/
│
├── app.py                # Main Streamlit application
├── requirements.txt      # List of dependencies
├── data/                 # Folder for saving CSV logs
└── README.md             # Documentation file
```

---

## 🖥️ Usage  

1. **Launch the app**: Use the `streamlit run app.py` command.  
2. **Upload a video or stream**: Start live video processing using the interface.  
3. **View results**: Detected license plates and extracted text will appear on the screen.  
4. **Export Data**: The license plates are logged into a CSV file in the `data/` folder for future reference.

---

## 🧩 Dependencies  
All required Python libraries are listed in the `requirements.txt` file:  
```bash
streamlit==1.32.0
ultralytics==8.2.0
opencv-python-headless==4.9.0.80
Pillow==10.3.0
easyocr==1.7.1
pandas==2.2.1
numpy==1.26.4
streamlit-webrtc==0.45.0
av==11.0.0
```

---

## 📝 License  
This project is licensed under the MIT License. You are free to use, modify, and distribute it.  

---

## 🙌 Acknowledgments  
- **Ultralytics** for YOLOv8  
- **JaidedAI** for EasyOCR  
- **Streamlit Community** for an intuitive Python web framework  

---

## 🤝 Contributing  
Contributions are welcome! If you'd like to add a feature or fix a bug, please fork the repository and submit a pull request.

---

## 📧 Contact  
For any inquiries or suggestions, feel free to reach out:  
**Name**: [Chelly Yassine]  
**Email**: [Yassinechelly04@gmail.com]  
**Linkedin**: [https://www.linkedin.com/in/yassine-chelly/]  

---
