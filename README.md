# Real-Time Facial Emotion Recognition using DeepFace & OpenCV

This project implements real-time facial emotion detection using Deep Learning.  
It uses the **DeepFace** framework for emotion analysis and **OpenCV Haar Cascade** for fast face detection.  
The system works in two modes:

### 📸 Image Mode  
Detect emotion from a given image.

### 🎥 Webcam Mode  
Real-time emotion recognition from your camera.

---

## 🙂 Detected Emotions  
happy | sad | neutral | angry | surprise | fear | disgust

---

## 🚀 Features
- Real-time webcam emotion recognition  
- Single-image emotion detection  
- Saves output image with bounding box + emotion label  
- Uses Mini-Xception CNN (pre-trained on FER-2013, CK+, JAFFE)  
- Lightweight and easy to run  

---

## 📦 Tech Stack
- Python  
- OpenCV  
- DeepFace  
- TensorFlow / tf_keras  

---

## 📁 Project Files
```
emotion.py  
requirements.txt  
README.md  
haarcascade_frontalface_default.xml  
```

---

## 📥 Installation
Install dependencies:

```
pip install -r requirements.txt
```

---

## ▶️ Usage

### **Image Mode**
```
python emotion.py
```
Enter:
```
image
```
Then type the image file name.

### **Webcam Mode**
```
python emotion.py
```
Enter:
```
camera
```
Press **Q** to quit webcam mode.

---

## 💾 Output
Processed images are saved automatically as:

```
output_emotion.jpg
```

---

## 👨‍💻 Author
**Priyansh Gautam (

