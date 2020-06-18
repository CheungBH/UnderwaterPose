#-*- coding: utf-8 -*-

from human_detection import ImgProcessor
import cv2
from config.config import video_path, frame_size
import numpy as np

body_parts = ["Nose", "Left eye", "Right eye", "Left ear", "Right ear", "Left shoulder", "Right shoulder", "Left elbow",
              "Right elbow", "Left wrist", "Right wrist", "Left hip", "Right hip", "Left knee", "Right knee",
              "Left ankle", "Right ankle"]
body_dict = {name: idx for idx, name in enumerate(body_parts)}

IP = ImgProcessor()
fourcc = cv2.VideoWriter_fourcc(*'XVID')
enhance_kernel = np.array([[0, -1, 0], [0, 5, 0], [0, -1, 0]])


class DrownDetector:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=200, detectShadows=False)
        self.height, self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    def process_video(self):
        cnt = 0
        while True:
            ret, frame = self.cap.read()
            frame = cv2.resize(frame, frame_size)
            cnt += 1
            if ret:
                fgmask = self.fgbg.apply(frame)
                background = self.fgbg.getBackgroundImage()
                diff = cv2.absdiff(frame, background)
                enhanced = cv2.filter2D(diff, -1, enhance_kernel)
                kps, img, black_img, boxes, kps_score = IP.process_img(frame, enhanced)
                cv2.imshow("res", img)
                cv2.imshow("res_black", black_img)
                cv2.waitKey(1)
            else:
                self.cap.release()
                break


if __name__ == '__main__':
    DrownDetector(video_path).process_video()
