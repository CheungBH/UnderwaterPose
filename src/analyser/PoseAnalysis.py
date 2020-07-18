from .box import Box
import cv2
import numpy as np
from .keypoint import Keypoint
from utils.kp_process import KPSProcessor
from src.human_detection import ImgProcessor
from collections import defaultdict
from src.classifymodel.TCN.test_TCN import TCNPredictor
from config.config import classifymodel,classifyframe,cls
from config import config

IP = ImgProcessor()

class Pose_Analysis:
    def __init__(self,height, width):
        self.KPSP = KPSProcessor(height, width)
        self.kps_dict = defaultdict(list)
        self.prediction = TCNPredictor(classifymodel, len(cls))
        self.pred = defaultdict(str)
        self.pred_dict = defaultdict(list)
        self.drown_list = 0
        self.coord = []
        self.cnt = 2
        self.num = 0

    def Analysis(self, kps, kps_score,frame,res):
        for key, v in kps.items():
            coord = self.KPSP.process_kp(v)
            self.kps_dict[key].append(coord)
        self.detect_kps()
        img, black_img = IP.visualize(kps,kps_score,frame)
        img = self.put_pred(img)
        return img, black_img

    def detect_kps(self):
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

    def put_pred(self, img):
        for idx, (k, v) in enumerate(self.pred.items()):
            cv2.putText(img, "id{}: {}".format(k,v), (30, int(40*(idx+1))), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            self.trigger_redalram(v,img)
        return img

    def trigger_redalram(self,value,img):
        if value == cls[1]:
            self.drown_list += 1
        else:
            self.drown_list = 0
        if self.drown_list >= 10:
            if self.num%self.cnt == 0:
                cv2.putText(img, "HELP!!!!", (360, 270), cv2.FONT_HERSHEY_SIMPLEX, 3,
                        (0, 0, 255), 3)
            self.num += 1

    def imageconcate(self,img,black_img,res):
        img = cv2.resize(img, (config.frame_size[0], config.frame_size[1]))
        black_img = cv2.resize(black_img, (config.frame_size[0], config.frame_size[1]))
        pose_res = np.concatenate((black_img, img), axis=1)
        image = np.vstack((res, pose_res))
        return image

    def clear(self):
        self.BOX = Box()
        self.KPS = Keypoint()
        self.disappear = 0

