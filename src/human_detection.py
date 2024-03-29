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

try:
    from config.config import gray_yolo_cfg, gray_yolo_weights, black_yolo_cfg, black_yolo_weights, video_path, \
        black_box_threshold, gray_box_threshold, pose_cfg, pose_weight
except:
    from src.debug.config.cfg_multi_detections import gray_yolo_cfg, gray_yolo_weights, black_yolo_cfg,\
        black_yolo_weights, video_path, black_box_threshold, gray_box_threshold, pose_cfg, pose_weight

fourcc = cv2.VideoWriter_fourcc(*'XVID')
empty_tensor = torch.empty([0,7])
empty_tensor4 = torch.empty([0,4])


class ImgProcessor:
    def __init__(self, show_img=True):
        self.black_yolo = ObjectDetectionYolo(cfg=black_yolo_cfg, weight=black_yolo_weights)
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
        self.RP = RegionProcessor(config.frame_size[0], config.frame_size[1], 10, 10)
        self.HP = HumanProcessor(config.frame_size[0], config.frame_size[1])
        self.BE = BoxEnsemble()
        self.kps = {}
        self.kps_score = {}

    def init(self):
        self.RP = RegionProcessor(config.frame_size[0], config.frame_size[1], 10, 10)
        self.HP = HumanProcessor(config.frame_size[0], config.frame_size[1])
        self.object_tracker = ObjectTracker()
        self.object_tracker.init_tracker()

    def process_img(self, frame, background):
        rgb_kps, dip_img, track_pred, rd_box = \
            copy.deepcopy(frame), copy.deepcopy(frame), copy.deepcopy(frame), copy.deepcopy(frame)
        img_black = cv2.imread("src/black.jpg")
        img_black = cv2.resize(img_black, config.frame_size)
        iou_img, black_kps, img_size_ls, img_box_ratio, rd_cnt = copy.deepcopy(img_black), \
            copy.deepcopy(img_black), copy.deepcopy(img_black), copy.deepcopy(img_black), copy.deepcopy(img_black)

        black_boxes, black_scores, gray_boxes, gray_scores = empty_tensor, empty_tensor, empty_tensor, empty_tensor
        diff = cv2.absdiff(frame, background)

        dip_boxes = self.dip_detection.detect_rect(diff)
        dip_results = [dip_img, dip_boxes]

        with torch.no_grad():
            # black picture
            enhance_kernel = np.array([[0, -1, 0], [0, 5, 0], [0, -1, 0]])
            enhanced = cv2.filter2D(diff, -1, enhance_kernel)
            black_res = self.black_yolo.process(enhanced)
            if black_res is not None:
                black_boxes, black_scores = self.black_yolo.cut_box_score(black_res)
                self.BBV.visualize(black_boxes, enhanced, black_scores)
                black_boxes, black_scores, black_res = \
                    filter_box(black_boxes, black_scores, black_res, black_box_threshold)
            black_results = [enhanced, black_boxes, black_scores]

            # gray pics process
            gray_img = gray3D(frame)
            gray_res = self.gray_yolo.process(gray_img)
            if gray_res is not None:
                gray_boxes, gray_scores = self.gray_yolo.cut_box_score(gray_res)
                self.BBV.visualize(gray_boxes, gray_img, gray_scores)
                gray_boxes, gray_scores, gray_res = \
                    filter_box(gray_boxes, gray_scores, gray_res, gray_box_threshold)
            gray_results = [gray_img, gray_boxes, gray_scores]

            merged_res = self.BE.ensemble_box(black_res, gray_res)

            self.id2bbox = self.object_tracker.track(merged_res)
            self.id2bbox = eliminate_nan(self.id2bbox)
            boxes = self.object_tracker.id_and_box(self.id2bbox)
            self.IDV.plot_bbox_id(self.id2bbox, track_pred, color=("red", "purple"), with_bbox=True)
            self.IDV.plot_bbox_id(self.object_tracker.get_pred(), track_pred, color=("yellow", "orange"), id_pos="down",
                                  with_bbox=True)

            self.object_tracker.plot_iou_map(iou_img)
            img_box_ratio = paste_box(rgb_kps, img_box_ratio, boxes)
            self.HP.update(self.id2bbox)

            self.RP.process_box(boxes, rd_box, rd_cnt)
            warning_idx = self.RP.get_alarmed_box_id(self.id2bbox)
            danger_idx = self.HP.box_size_warning(warning_idx)
            self.HP.vis_box_size(img_box_ratio, img_size_ls)

            if danger_idx:
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
                                self.RNN_model.vis_RNN_res(n, idx, self.HP.get_RNN_preds(idx), black_kps)

            detection_map = np.concatenate((enhanced, gray_img), axis=1)
            tracking_map = np.concatenate((track_pred, iou_img), axis=1)
            row_1st_map = np.concatenate((detection_map, tracking_map), axis=1)
            box_map = np.concatenate((img_box_ratio, img_size_ls), axis=1)
            rd_map = np.concatenate((rd_cnt, rd_box), axis=1)
            row_2nd_map = np.concatenate((rd_map, box_map), axis=1)
            kps_map = np.concatenate((rgb_kps, black_kps), axis=1)
            cache_map = np.concatenate((frame, img_black), axis=1)
            row_3rd_map = np.concatenate((kps_map, cache_map), axis=1)
            res_map = np.concatenate((row_1st_map, row_2nd_map, row_3rd_map), axis=0)

        return gray_results, black_results, dip_results, res_map
