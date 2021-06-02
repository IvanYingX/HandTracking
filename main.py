import cv2
import mediapipe as mp
import time
from HandTracking import HandTracking

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
time_0 = 0
hand = HandTracking()
i = 0
while True:
    time_c = time.time()
    f = 1 / (time_c - time_0)
    time_0 = time_c
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = hand.detect_hand(img)
    lm_list = hand.get_positions(img)
    if lm_list:
        print(lm_list)

    else:
        print('Not found')
    cv2.putText(img, 'fps = ' + str(int(f)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    cv2.imshow('Image', img)

    if cv2.waitKey(1) == ord('q'):
        break