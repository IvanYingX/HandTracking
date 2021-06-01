import cv2
import mediapipe as mp
import time
from HandTracking import HandTracking
import numpy as np
from subprocess import call


cap = cv2.VideoCapture(0)
time_0 = 0
hand = HandTracking(hands=1)
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
        index_p = np.array([lm_list[8][1], lm_list[8][2]])
        thumb_p = np.array([lm_list[4][1], lm_list[4][2]])
        cv2.line(img, index_p, thumb_p, (0, 0, 150), 3)
        mid_x, mid_y =  (index_p[0] + thumb_p[0]) // 2, (index_p[1] + thumb_p[1]) // 2  
        cv2.circle(img, (mid_x, mid_y), 15, (0, 0, 150), cv2.FILLED)
        distance = np.sqrt(np.sum((index_p - thumb_p)**2))
        volume = np.interp(distance, [50, 230], [0, 100])
        call(["amixer", "-D", "pulse", "sset", "Master", str(volume)+"%"])
        # index_x, index_y = lm_list[8][1], lm_list[8][2]
        # thumb_x, thumb_y = lm_list[4][1], lm_list[4][2]
        # mid_x, mid_y =  (index_x + thumb_x) // 2, (index_y + thumb_y) // 2  
        # cv2.line(img, (thumb_x, thumb_y),
        #          (index_x, index_y), (0, 0, 150), 3)
        # cv2.circle(img, (mid_x, mid_y), 15, (0, 0, 150), cv2.FILLED)
    cv2.putText(img, 'fps = ' + str(int(f)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    cv2.imshow('Image', img)

    if cv2.waitKey(1) == ord('q'):
        break