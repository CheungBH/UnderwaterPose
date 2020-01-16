from .opt import opt
from SPPE.src.utils.eval import getPrediction
import numpy as np

from .visualizer import KeyPointVisualizer
from .nms import pose_nms


class PoseEstimator(object):
    def __init__(self):
        self.skeleton = []
        self.KPV = KeyPointVisualizer()

    def process(self, boxes, scores, hm_data, pt1, pt2, orig_img, im_name):
        orig_img = np.array(orig_img, dtype=np.uint8)
        if boxes is None:
            return orig_img, [], [], boxes
        else:
            # location prediction (n, kp, 2) | score prediction (n, kp, 1)
            preds_hm, preds_img, preds_scores = getPrediction(
                hm_data, pt1, pt2, opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW)
            kps, score = pose_nms(boxes, scores, preds_img, preds_scores)

            if kps:
                img_black = self.KPV.vis_ske_black(orig_img, kps, score)
                img = self.KPV.vis_ske(orig_img, kps, score)
                return img, kps, img_black, boxes
            else:
                return orig_img, [], orig_img, boxes

