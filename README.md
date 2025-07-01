# AI-Assistant-for-Visually-Impaired
#Haleema Iftikhar
Contact: haleema4work@gmail.com
🔍Summary: Real-Time Object and Emotion Detection System for Assistive Vision Applications
Author: Haleema Iftikhar
Project Title: Real-Time Visual Perception System for Visually Impaired Users Using Deep Learning and On-Device Audio Feedback
Technologies: OpenCV, ONNX Runtime, pyttsx3, MobileNet-SSD, FER+, Haar cascades
Objectives:
1) Detect and classify everyday objects in the user’s environment.

2)Estimate the direction and relative distance of detected objects.

3)Recognize human emotions based on facial expressions.

4)Provide natural language feedback through real-time text-to-speech (TTS).

5)Operate in real time and requires minimal compute resources.
System Architecture & Methodology:
•	Object Detection:
The system uses a pre-trained MobileNet-SSD model trained on the COCO dataset to detect 80 common object classes. Detected objects are analyzed based on bounding box geometry to infer:
Direction (left, center, right) based on x-coordinate center
Distance (very close, medium, far) based on bounding box height
•	Face & Emotion Detection:
When a "person" is detected, the system performs:
Face localization using Haar cascades
Grayscale normalization and resizing to 64×64 resolution

Inference using the FER+ ONNX model for 8 emotion categories:
['neutral','happiness','surprise','sadness','anger','disgust','fear','contempt']
Emotion is only reported if its confidence probability exceeds a defined threshold (default: 40%).
•	Speech Output:
Pyttsx3 is used for text-to-speech conversion. Spoken messages are throttled using a temporal buffer to avoid information overload.
Features:
📦 Object Localization with verbal direction & proximity

😊 Emotion Recognition for social awareness

🧠 ONNX Inference for hardware-efficient deep learning

🔊 Voice Feedback to aid non-visual navigation

⚡ Real-Time Performance on CPU-only systems

Applications:
•	Assistive technology for blind or low-vision individuals

•	Human–robot interaction systems requiring emotion understanding

•	Context-aware wearable devices

•	Ambient intelligence in smart home or public environments

Limitations & Future Improvements:
•	Emotion recognition is limited by FER+ model accuracy and lighting conditions.

•	The current system supports a single-face emotion analysis.

•	Future versions may include:

•	Multilingual speech support

•	Depth estimation via stereo or IR

•	Gesture or action recognition

•	Integration with GPS or haptic feedback
