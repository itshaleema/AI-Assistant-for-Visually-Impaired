import cv2
import pyttsx3
import time

engine = pyttsx3.init()
cap = cv2.VideoCapture("http://192.168.1.4:8080/video")
cap.set(3, 640)
cap.set(4, 480)

# Load classes
with open("coco.names", 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load model
configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Speech throttle
last_spoken = 0
SPEAK_GAP = 1.0  # seconds

while True:
    success, img = cap.read()
    if not success:
        break

    classIds, confs, bbox = net.detect(img, confThreshold=0.5)
    h, w = img.shape[:2]
    to_speak = []

    if len(classIds):
        for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
            x, y, bw, bh = box
            label = classNames[classId - 1].upper()

            # Direction based on x-center
            cx = x + bw // 2
            if cx < w // 3:
                direction = "left"
            elif cx > 2 * w // 3:
                direction = "right"
            else:
                direction = "center"

            # Distance estimation based on height
            if bh > h * 0.5:
                distance = "very close"
            elif bh > h * 0.25:
                distance = "medium"
            else:
                distance = "far"

            # Text on frame
            text = f"{label} ({direction}, {distance})"
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            to_speak.append(f"{label.lower()} {distance} on the {direction}")

    # Speak after a gap
    now = time.time()
    if to_speak and now - last_spoken >= SPEAK_GAP:
        engine.say("; ".join(to_speak))
        engine.runAndWait()
        last_spoken = now

    # Resize for better display
    resized_img = cv2.resize(img, (800, 600))  # can change to (1024, 768) or higher
    cv2.imshow("Blind-Aid", resized_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
