import torch
import numpy as np
import cv2
import copy
from config import config
from src.detector.yolo_detect import ObjectDetectionYolo
from src.detector.image_process_detect import ImageProcessDetection
# from src.detector.yolo_asff_detector import ObjectDetectionASFF
from src.detector.visualize import BBoxVisualizer
from src.estimator.pose_estimator import PoseEstimator
from src.estimator.visualize import KeyPointVisualizer
from src.utils.img import gray3D
from src.detector.box_postprocess import crop_bbox, filter_box, BoxEnsemble, eliminate_nan
from src.tracker.track import ObjectTracker
from src.tracker.visualize import IDVisualizer
from src.analyser.area import RegionProcessor
from src.analyser.humans import HumanProcessor
from src.utils.utils import paste_box
from src.RNNclassifier.classify import RNNInference
from config.config import gray_yolo_cfg, gray_yolo_weights, gray_box_threshold, pose_cfg, pose_weight
if config.research:
    from config.config import black_yolo_cfg,black_yolo_weights,black_box_threshold
from server_test import server
from utils.alarm import *
empty_tensor = torch.empty([0,7])
torch.cuda.set_device(0)

class ImgProcessor:
    def __init__(self, resize_size, show_img=True):
        self.gray_yolo = ObjectDetectionYolo(cfg=gray_yolo_cfg, weight=gray_yolo_weights)
        self.object_tracker = ObjectTracker()
        self.dip_detection = ImageProcessDetection()
        self.RNN_model = RNNInference()
        self.pose_estimator = PoseEstimator(pose_cfg=pose_cfg, pose_weight=pose_weight)
        self.KPV = KeyPointVisualizer()
        self.BBV = BBoxVisualizer()
        self.IDV = IDVisualizer()
        self.img = []
        self.id2bbox = {}
        self.img_black = []
        self.show_img = show_img
        self.RP = RegionProcessor(resize_size[0], resize_size[1], 10, 10)
        self.HP = HumanProcessor(resize_size[0], resize_size[1])
        self.BE = BoxEnsemble(resize_size[0], resize_size[1])
        self.kps = {}
        self.kps_score = {}
        self.S = server()
        self.resize_size = resize_size
        # self.ser = serial.Serial(
        #     port="/dev/ttyUSB0",
        #     baudrate=9600#,
        # )
        # print(self.ser.isOpen())
        self.red_on = [0xA0,0x01,0x01,0xA2]
        self.yellow_on = [0xA0,0x02,0x01,0xA3]
        self.green_on = [0xA0,0x03,0x01,0xA4]
        self.buzzer_on = [0xA0,0x04,0x01,0xA5]
        self.red_off = [0xA0,0x01,0x00,0xA1]
        self.yellow_off = [0xA0,0x02,0x00,0xA2]
        self.green_off = [0xA0,0x03,0x00,0xA3]
        self.buzzer_off = [0xA0,0x04,0x00,0xA4]
        self.signal = 0
        self.alrambox = []

        if config.research:
            self.black_yolo = ObjectDetectionYolo(cfg=black_yolo_cfg, weight=black_yolo_weights)

    def init(self):
        self.RP = RegionProcessor(self.resize_size[0], self.resize_size[1], 10, 10)
        self.HP = HumanProcessor(self.resize_size[0], self.resize_size[1])
        self.object_tracker = ObjectTracker()
        self.object_tracker.init_tracker()


    def process_img(self, frame, background):
        rgb_kps, dip_img, rd_box = \
            copy.deepcopy(frame), copy.deepcopy(frame), copy.deepcopy(frame)
        img_black = cv2.resize(cv2.imread("src/black.jpg"), self.resize_size)
        black_kps, img_box_ratio, rd_cnt = copy.deepcopy(img_black), \
            copy.deepcopy(img_black), copy.deepcopy(img_black)

        gray_boxes, gray_scores = empty_tensor, empty_tensor
        diff = cv2.absdiff(frame, background)
        dip_boxes = self.dip_detection.detect_rect(diff)
        dip_results = [dip_img, dip_boxes]

        with torch.no_grad():
            # gray pics process
            gray_img = gray3D(frame)
            gray_res = self.gray_yolo.process(gray_img)
            if gray_res is not None:
                gray_boxes, gray_scores = self.gray_yolo.cut_box_score(gray_res)
                gray_boxes, gray_scores, gray_res = \
                    filter_box(gray_boxes, gray_scores, gray_res, gray_box_threshold)
                self.BBV.visualize(gray_boxes, gray_img, gray_scores)

            gray_results = [gray_img, gray_boxes, gray_scores]
            merged_res = gray_res
            self.id2bbox = self.object_tracker.track(merged_res)
            self.id2bbox = eliminate_nan(self.id2bbox)
            boxes = self.object_tracker.id_and_box(self.id2bbox)
            img_box_ratio = paste_box(rgb_kps, img_box_ratio, boxes)
            self.HP.update(self.id2bbox)
            self.RP.process_box(boxes, rd_box, rd_cnt)
            warning_idx = self.RP.get_alarmed_box_id(self.id2bbox)
            danger_idx = self.HP.box_size_warning(warning_idx) #After ratio of the bounding box

            if danger_idx:
                # if self.signal == 0:
                    # self.S.connect(1,1)
                    # self.ser.write(serial.to_bytes(self.green_off))
                    # self.ser.write(serial.to_bytes(self.red_off))
                    # self.ser.write(serial.to_bytes(self.yellow_on))
                    # self.ser.write(serial.to_bytes(self.buzzer_off))
                danger_id2box = {k:v for k,v in self.id2bbox.items() if k in danger_idx}
                danger_box = self.object_tracker.id_and_box(danger_id2box)
                inps, pt1, pt2 = crop_bbox(rgb_kps, danger_box)

                if inps is not None:
                    kps, kps_score, kps_id = self.pose_estimator.process_img(inps, danger_box, pt1, pt2)
                    if self.kps is not []:
                        self.kps, self.kps_score = self.object_tracker.match_kps(kps_id, kps, kps_score)
                        self.HP.update_kps(self.kps)
                        self.KPV.vis_ske(rgb_kps, kps, kps_score)
                        self.IDV.plot_bbox_id(danger_id2box, rgb_kps, with_bbox=True)
                        self.IDV.plot_skeleton_id(self.kps, rgb_kps)
                        self.KPV.vis_ske_black(black_kps, kps, kps_score)
                        self.IDV.plot_skeleton_id(self.kps, black_kps)

                        for n, idx in enumerate(self.kps.keys()):
                            if self.HP.if_enough_kps(idx):
                                RNN_res = self.RNN_model.predict_action(self.HP.obtain_kps(idx))
                                self.HP.update_RNN(idx, RNN_res)
                                if self.HP.get_RNN_preds(idx)[0] == 'drown' and len(set(self.HP.get_RNN_preds(idx))) == 1 \
                                        and len(self.HP.get_RNN_preds(idx)) == 5:

                                    self.alrambox.append(danger_id2box.get(idx).tolist()[0] / black_kps.shape[1])
                                    self.alrambox.append(danger_id2box.get(idx).tolist()[1] / black_kps.shape[0])
                                    self.alrambox.append(danger_id2box.get(idx).tolist()[2] / black_kps.shape[1])
                                    self.alrambox.append(danger_id2box.get(idx).tolist()[3] / black_kps.shape[0])
                                    self.signal = 2
                                    # self.ser.write(serial.to_bytes(self.yellow_off))
                                    # self.ser.write(serial.to_bytes(self.red_on))
                                    # self.S.connect(2,str(self.alrambox)[1:-1])
                                    # self.ser.write(serial.to_bytes(self.buzzer_on))
                                elif self.HP.get_RNN_preds(idx)[0] == 'stand' and len(set(self.HP.get_RNN_preds(idx))) == 1 \
                                        and len(self.HP.get_RNN_preds(idx)) == 5:
                                    self.signal = 0
                                    # self.S.connect(0,0)
                                self.RNN_model.vis_RNN_res(n, idx, self.HP.get_RNN_preds(idx), black_kps)
                                self.alrambox = []
            else:
                # self.ser.write(serial.to_bytes(self.red_off))
                # self.ser.write(serial.to_bytes(self.yellow_off))
                # self.ser.write(serial.to_bytes(self.buzzer_off))
                # self.ser.write(serial.to_bytes(self.green_on))
                self.signal = 0
                # self.S.connect(0,0)

            row_1st_map = np.concatenate((gray_img, rd_box), axis=1)
            row_2nd_map = np.concatenate((img_box_ratio, black_kps), axis=1)
            res_map = np.concatenate((row_1st_map, row_2nd_map), axis=0)
            sent = cv2.resize(res_map, (320, 240))
            # self.S.connect(sent.data, 0)

        return gray_results, dip_results, res_map
