import cv2
import mediapipe as mp
import time
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    h, w, c = img.shape
    
    
    
            # wrist = handLMs.landmark[0]
            # cx, cy = int(wrist.x * w), int(wrist.y * h)
            # cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

    time_c = time.time()
    f = 1 / (time_c - time_0)
    time_0 = time_c
    cv2.putText(img, str(int(f)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    cv2.imshow('Image', img)
    cv2.waitKey(1)