# Real-Time Object and Emotion Detection System for Assistive Vision Applications
### 👩‍💻 Author: Haleema Iftikhar  
📧 **Contact:** haleema4work@gmail.com  

---

## 🔍 Summary
**Project Title:** *Real-Time Visual Perception System for Visually Impaired Users Using Deep Learning and On-Device Audio Feedback*  
**Technologies Used:** OpenCV, ONNX Runtime, pyttsx3, MobileNet-SSD, FER+, Haar Cascades  

---

## 🎯 Objectives
1. Detect and classify everyday objects in the user’s environment  
2. Estimate the **direction** and **relative distance** of detected objects  
3. Recognize **human emotions** based on facial expressions  
4. Provide **natural language feedback** using real-time text-to-speech (TTS)  
5. Operate in **real time** with minimal compute resources  

---

## 🛠 System Architecture & Methodology

### • Object Detection
- Uses **MobileNet-SSD** trained on the COCO dataset (80 object classes)
- Computes:
  - **Direction**: left, center, right (based on x-center of bounding box)
  - **Distance**: very close, medium, far (based on bounding box height)

### • Face & Emotion Detection
- Triggered when a `"person"` is detected
- Pipeline:
  - Face localized using **Haar Cascades**
  - Grayscale normalized and resized to **64×64**
  - Inference via **FER+ ONNX** model
- Supported emotion categories:
  - `['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']`
- Emotion is only reported if confidence > **40%**

### • Speech Output
- Uses **pyttsx3** for offline TTS
- Voice feedback is throttled to avoid cognitive overload

---

## 🌟 Key Features
- 📦 **Object Localization** with verbal direction & proximity
- 😊 **Emotion Recognition** for social interaction
- 🧠 **ONNX Runtime** for efficient on-device inference
- 🔊 **Real-Time Voice Feedback** for non-visual navigation
- ⚡ Runs efficiently on **CPU-only systems**

---

## 🧩 Applications
- Assistive tech for **blind or visually impaired users**
- Emotion-aware **human–robot interaction**
- **Smart wearable devices** with contextual awareness
- **Ambient intelligence** in smart homes or public spaces

---

## ⚠️ Limitations & Future Work
- Emotion recognition depends on FER+ accuracy and lighting
- Currently supports **only one face** at a time
- Future enhancements:
  - 🌐 **Multilingual TTS**
  - 🕹️ **Gesture or action recognition**
  - 🧭 **GPS or haptic feedback integration**
  - 👓 Support for **depth sensors** (IR/stereo vision)

---
Thank you!!
