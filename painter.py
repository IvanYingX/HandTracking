import cv2
import mediapipe as mp
from HandTracking import HandTracking
import numpy as np
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 720)

hand = HandTracking(hands=1)
xp = np.array([0, 0])
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = hand.detect_hand(img)
    lm_list = hand.get_positions(img)
    if lm_list:
        fingers = hand.fingers_up(img, lm_list)
        index_tip = np.array([lm_list[8][1], lm_list[8][2]])
        # thumb_tip = np.array([lm_list[4][1], lm_list[4][2]])
        # ring_tip = np.array([lm_list[12][1], lm_list[12][2]])
        
        # dist = np.sqrt(np.sum((index_tip - thumb_tip)**2))
        if xp[0] == 0 and xp[1] == 0:
            xp = index_tip 
        if fingers[1] and fingers[2] == 0:
            cv2.line(imgCanvas, xp, index_tip, (255, 0, 0), 3) 
        cv2.line(img, xp, index_tip, (255, 0, 0), 3) 
        xp = index_tip
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 20, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)
    # img = cv2.addWeighted(img, 0.9, imgCanvas, 0.1, 0)
    cv2.imshow('Image', img)
    cv2.imshow('Canvas', imgCanvas)
    if cv2.waitKey(1) == ord('q'):
        break