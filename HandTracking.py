import cv2
import mediapipe as mp
import time

class HandTracking:
    def __init__(self, mode=False, hands=2, detect_conf=0.5, tracking_conf=0.5):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(mode, hands, detect_conf, tracking_conf)
        self.mpDraw = mp.solutions.drawing_utils

    def detect_hand(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLMs in results.multi_hand_landmarks:
                if draw:  
                    self.mpDraw.draw_landmarks(img,
                                           handLMs,
                                           self.mpHands.HAND_CONNECTIONS)
        return img

    def identify_point(self, img, point=0, hand=0, draw=True):
        h, w, c = img.shape
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        lm_list = []
        if results.multi_hand_landmarks:
            my_hand = results.multi_hand_landmarks[hand]
            lm = my_hand.landmark[point]
            cx, cy = int(lm.x * w), int(lm.y * h)
            lm_list = [cx, cy]
            cv2.circle(img, (lm_list[0], lm_list[1]),
                   10, (255, 0, 255), cv2.FILLED)
        return lm_list