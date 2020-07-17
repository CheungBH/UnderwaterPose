#-*- coding: utf-8 -*-
from collections import defaultdict
from src.human_detection_new import ImgProcessor
import cv2
from config.config import video_path, frame_size,classifymodel,classifyframe,cls
import numpy as np
from utils.kp_process import KPSProcessor
from src.classifymodel.TCN.test_TCN import TCNPredictor

body_parts = ["Nose", "Left eye", "Right eye", "Left ear", "Right ear", "Left shoulder", "Right shoulder", "Left elbow",
              "Right elbow", "Left wrist", "Right wrist", "Left hip", "Right hip", "Left knee", "Right knee",
              "Left ankle", "Right ankle"]
body_dict = {name: idx for idx, name in enumerate(body_parts)}

IP = ImgProcessor()
fourcc = cv2.VideoWriter_fourcc(*'XVID')
enhance_kernel = np.array([[0, -1, 0], [0, 5, 0], [0, -1, 0]])


class DrownDetector:
    def __init__(self, video_path, classifymodel):
        self.cap = cv2.VideoCapture(video_path)
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=200, detectShadows=False)
        self.height, self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.KPSP = KPSProcessor(self.height, self.width)
        self.kps_dict = defaultdict(list)
        self.coord = []
        self.prediction = TCNPredictor(classifymodel,2)
        self.pred = defaultdict(str)
        self.pred_dict = defaultdict(list)

    def __detect_kps(self):
        refresh_idx = []
        for k, v in self.kps_dict.items():
            if len(v) == classifyframe:
                pred = self.prediction.predict(np.array(v).astype(np.float32))
                self.pred[k] = cls[pred]
                self.pred_dict[str(k)].append(cls[pred])
                # print("Predicting id {}".format(k))
                refresh_idx.append(k)
        for idx in refresh_idx:
            self.kps_dict[idx] = []

    def __put_pred(self, img):
        for idx, (k, v) in enumerate(self.pred.items()):
            cv2.putText(img, "id{}: {}".format(k,v), (30, int(40*(idx+1))), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        return img

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
                kps, boxes, kps_score = IP.process_img(frame, enhanced)
                if kps:
                    for key, v in kps.items():
                        # coord = self.__normalize_coordinates(kps[key])
                        coord = self.KPSP.process_kp(v)
                        self.kps_dict[key].append(coord)
                    self.__detect_kps()
                img, black_img = IP.visualize()
                img = self.__put_pred(img)
                cv2.imshow("res", img)
                #cv2.imshow("res_black", black_img)
                cv2.waitKey(1)
            else:
                self.cap.release()
                break


if __name__ == '__main__':
    DrownDetector(video_path,classifymodel).process_video()
