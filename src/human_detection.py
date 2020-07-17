import torch
import cv2
import copy
import numpy as np
from config import config
from src.estimator.pose_estimator import PoseEstimator
from src.estimator.visualize import KeyPointVisualizer
from src.detector.yolo_detect import ObjectDetectionYolo
from src.detector.visualize import BBoxVisualizer
from src.tracker.track import ObjectTracker
from src.tracker.visualize import IDVisualizer
from src.utils.img import torch_to_im, gray3D
from src.analyser.area import RegionProcessor
from src.detector.box_postprocess import crop_bbox, merge_box

try:
    from config.config import gray_yolo_cfg, gray_yolo_weights, black_yolo_cfg, black_yolo_weights, \
        video_path, pose_cfg, pose_weight
except:
    from src.debug.config.cfg_multi_detections import gray_yolo_cfg, gray_yolo_weights, black_yolo_cfg, \
        black_yolo_weights, video_path, pose_cfg, pose_weight

tensor = torch.FloatTensor


class ImgProcessor:
    def __init__(self, show_img=False):
        self.gray_detector = ObjectDetectionYolo(cfg=gray_yolo_cfg, weight=gray_yolo_weights)
        self.black_detector = ObjectDetectionYolo(cfg=black_yolo_cfg, weight=black_yolo_weights)
        self.pose_estimator = PoseEstimator(pose_cfg=pose_cfg, pose_weight=pose_weight)
        self.object_tracker = ObjectTracker()
        self.BBV = BBoxVisualizer()
        self.KPV = KeyPointVisualizer()
        self.IDV = IDVisualizer(with_bbox=False)
        self.boxes = tensor([])
        self.boxes_scores = tensor([])
        self.img_black = np.array([])
        self.frame = np.array([])
        self.id2bbox = {}
        self.kps = {}
        self.kps_score = {}
        self.RP = RegionProcessor(config.frame_size[0], config.frame_size[1], 10, 10)
        self.show_img = show_img
        self.alarm_ls = []
        self.center_point = []
        self.boxesforpose = []
        self.boxepose = []
        self.item = 0

    def init_sort(self):
        self.object_tracker.init_tracker()

    def clear_res(self):
        self.boxes = tensor([])
        self.boxes_scores = tensor([])
        self.frame = np.array([])
        self.id2bbox = {}
        self.kps_score = {}
        self.center_point = []
        self.boxesforpose = []
        self.boxepose = []
        self.kps = {}

    def visualize(self,kps,kps_score,frame):
        self.kps = kps
        self.kps_score = kps_score
        self.frame = frame
        img_black = cv2.imread('../video/black.jpg')
        if config.plot_bbox and self.boxes is not None:
            self.frame = self.BBV.visualize(self.boxes, self.frame)
            # cv2.imshow("cropped", (torch_to_im(inps[0]) * 255))
        if config.plot_kps and self.kps is not []:
            self.frame = self.KPV.vis_ske(self.frame, self.kps, self.kps_score)
            img_black = self.KPV.vis_ske_black(self.frame, self.kps, self.kps_score)
        if config.plot_id and self.id2bbox is not None:
            self.frame = self.IDV.plot_bbox_id(self.id2bbox, self.frame)
            self.frame = self.IDV.plot_skeleton_id(self.kps, self.frame)

        return self.frame, img_black

    def get_centerpoint(self,region):
        for item in region:
            center = (int((item[0] + 0.5) * config.frame_size[0])/10,
                                 int((item[1] + 0.5) *config.frame_size[1])/10)
            if center not in self.center_point:
                self.center_point.append(center)


    def isInside(self, points, bbox, region):
        for item in bbox:
            for center in points:
                if region[center].center[0] <= item[2].item() and region[center].center[0] >= item[0].item() and region[center].center[1] <= item[3].item()\
                        and region[center].center[1] >= item[1].item():
                    item = item.unsqueeze(dim=0)
                    self.boxepose.append(item)
                    self.boxesforpose = torch.cat(self.boxepose,dim=0)
                    break


    def process_img(self, frame, black_img):
        self.clear_res()
        self.frame = frame
        res = cv2.resize(frame,(1440,540))


        with torch.no_grad():
            gray_img = gray3D(copy.deepcopy(frame))
            gray_results = self.gray_detector.process(gray_img)
            black_results = self.black_detector.process(black_img)

            gray_boxes, gray_scores = self.gray_detector.cut_box_score(gray_results)
            black_boxes, black_scores = self.black_detector.cut_box_score(black_results)

            self.boxes, self.boxes_scores = merge_box(gray_boxes, black_boxes, gray_scores, black_scores)

            if self.show_img:
                gray_img = self.BBV.visualize(gray_boxes, gray_img, gray_scores)
                cv2.imshow("gray", gray_img)
                black_img = self.BBV.visualize(black_boxes, black_img, black_scores)
                cv2.imshow("black", black_img)

            if gray_results is not None:
                self.id2bbox = self.object_tracker.track(gray_results)
                boxes = self.object_tracker.id_and_box(self.id2bbox)
                self.alarm_ls, REGIONS,res = self.RP.process_box(boxes, copy.deepcopy(frame))
                if self.alarm_ls:
                    self.isInside(self.alarm_ls, boxes, REGIONS)
                    if len(self.boxesforpose)>0:
                        inps, pt1, pt2 = crop_bbox(frame, self.boxesforpose)
                        if inps is not None:
                            kps, kps_score, kps_id = self.pose_estimator.process_img(inps, self.boxesforpose, pt1, pt2)
                            self.kps, self.kps_score = self.object_tracker.match_kps(kps_id, kps, kps_score)

        return self.kps, self.id2bbox, self.kps_score,self.frame, res