import cv2
import mediapipe as mp
import time

class HandTracking:
    def __init__(self, mode=False, hands=2, detect_conf=0.5, tracking_conf=0.5):
        mpHands = mp.solutions.hands
        self.hands = mpHands.Hands(mode, hands, detect_conf, tracking_conf)
        self.mpDraw = mp.solutions.drawing_utils

    def detect_hand(self, img, draw=True):
        time_0 = 0
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLMs in results.multi_hand_landmarks:
                if draw:  
                    self.mpDraw.draw_landmarks(img,
                                           handLMs,
                                           self.mpHands.HAND_CONNECTIONS)
