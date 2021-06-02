import cv2
import mediapipe as mp
from HandTracking import HandTracking

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
hand = HandTracking(hands=1, detect_conf=0.7, tracking_conf=0.7)
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = hand.detect_hand(img)
    lm_list = hand.get_positions(img)
    fingers = [0] * 5
    if lm_list:
        fingers = hand.fingers_up(img, lm_list)
    cv2.putText(img, str(sum(fingers)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    cv2.imshow('Image', img)

    if cv2.waitKey(1) == ord('q'):
        break