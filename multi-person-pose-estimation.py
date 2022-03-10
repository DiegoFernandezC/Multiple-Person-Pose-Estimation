# Classes ID
#   person: 0
#   sports ball: 32

# Video: 640x480

from ctypes import sizeof
import cv2
import numpy as np
import mediapipe as mp

# Yolo
net = cv2.dnn.readNet("../YOLO/yolov3.weights", "../YOLO/yolov3.cfg")
classes = []
with open("../YOLO/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

#Mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

pTime = 0

# Loading web cam
camera = cv2.VideoCapture(0)
# Loading picture
#img = cv2.imread("test_image.jpg")

while True:
    success, img = camera.read()
    height, width, channels = img.shape

    # Yolo to detect objects
    blob = cv2.dnn.blobFromImage(img, 1.0/255, (320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            if class_id == 0 or class_id == 32:
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)      

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            print(x, y, w, h)
            # Run MediaPipe pose estimator for each person detected
            if class_ids[i] == 0:
                crop_img = img[abs(y):abs(y)+h, abs(x):abs(x)+w]
                imgRGB = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                results = pose.process(crop_img)
                if results.pose_landmarks:
                    mpDraw.draw_landmarks(crop_img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
                    for id, lm in enumerate(results.pose_landmarks.landmark):
                        h, w,c = crop_img.shape
                        print(id, lm)
                        cx, cy = int(lm.x*w), int(lm.y*h)
                        cv2.circle(crop_img, (cx, cy), 5, (255,0,0), cv2.FILLED)
            #Box and label
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 15), font, 1.2, color, 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

#cv2.imwrite("result2.jpg", img)

camera.release()
cv2.destroyAllWindows()