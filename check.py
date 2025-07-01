#Haleema Iftikhar
import cv2
import pyttsx3
import time
import onnxruntime as ort
import numpy as np

# -------------- constants / tweakables --------------
FACE_PAD   = 10      # extra pixels around face crop
EMO_THRESH = 0.40    # speak emotion only if prob > 40 %
SPEAK_GAP  = 1.0     # seconds between spoken messages
# ----------------------------------------------------

engine = pyttsx3.init()

# --- camera (0 = laptop cam)
##cap = cv2.VideoCapture("http://192... ip /video")   # phone stream
cap = cv2.VideoCapture(0)
cap.set(3, 640); cap.set(4, 480)

# --- COCO class names ---
classNames = open("coco.names").read().strip().split('\n')

# --- MobileNet‑SSD object detector ---
net = cv2.dnn_DetectionModel("frozen_inference_graph.pb",
                             "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
net.setInputSize(320, 320)
net.setInputScale(1/127.5)
net.setInputMean((127.5,)*3)
net.setInputSwapRB(True)

# --- FER+ emotion model (ONNX) ---
sess = ort.InferenceSession("emotion-ferplus-8.onnx",
                            providers=["CPUExecutionProvider"])
IN_NAME  = sess.get_inputs()[0].name
OUT_NAME = sess.get_outputs()[0].name
emotions = ['neutral','happiness','surprise','sadness',
            'anger','disgust','fear','contempt']

# --- Haar face detector ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     "haarcascade_frontalface_default.xml")

last_spoken = 0

# ==================== MAIN LOOP ====================
while True:
    ok, frame = cap.read()
    if not ok:
        break

    H, W = frame.shape[:2]
    speech_queue = []

    # ---------- object detection ----------
    ids, confs, boxes = net.detect(frame, confThreshold=0.5)

    if len(ids):
        for cid, conf, box in zip(ids.flatten(), confs.flatten(), boxes):
            x, y, w, h = box
            label = classNames[cid-1].upper()

            # --- direction & distance ---
            cx = x + w // 2
            direction = "left" if cx < W//3 else "right" if cx > 2*W//3 else "center"
            distance  = "very close" if h > .5*H else "medium" if h > .25*H else "far"

            overlay = f"{label} ({direction}, {distance})"
            speak   = f"{label.lower()} {distance} on the {direction}"

            # ---------- emotion if PERSON ----------
            if label.lower() == "person":
                # grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(40,40)
                )

                if len(faces):
                    # pick largest face
                    fx, fy, fw, fh = max(faces, key=lambda f: f[2]*f[3])

                    # pad & clamp to frame bounds
                    px = max(fx - FACE_PAD, 0)
                    py = max(fy - FACE_PAD, 0)
                    pw = min(fw + 2*FACE_PAD, W - px)
                    ph = min(fh + 2*FACE_PAD, H - py)
                    face_gray = gray[py:py+ph, px:px+pw]

                    # 64×64, float32, **NO** /255 scaling
                    face_resized = cv2.resize(face_gray, (64, 64)).astype('float32')
                    blob = face_resized.reshape(1, 1, 64, 64)

                    # run FER+ model
                    logits = sess.run([OUT_NAME], {IN_NAME: blob})[0][0]
                    exp    = np.exp(logits - np.max(logits))
                    probs  = exp / exp.sum()

                    emo_idx = int(np.argmax(probs))
                    if probs[emo_idx] >= EMO_THRESH:
                        emo = emotions[emo_idx]
                        overlay += f" — {emo}"
                        speak   += f", looks {emo}"

                        # draw inner face box & label
                        cv2.rectangle(frame, (px, py), (px+pw, py+ph), (255, 0, 255), 1)
                        cv2.putText(frame, emo, (px, py-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            # ---------- draw object box & text ----------
            cv2.rectangle(frame, box, (0, 255, 0), 2)
            cv2.putText(frame, overlay, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            speech_queue.append(speak)

    # ---------- speech ----------
    now = time.time()
    if speech_queue and (now - last_spoken) >= SPEAK_GAP:
        engine.say("; ".join(speech_queue))
        engine.runAndWait()
        last_spoken = now

    # ---------- display ----------
    cv2.imshow("Blind‑Aid  (press q to quit)", cv2.resize(frame, (800, 600)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
