from estimator.pose_estimator import PoseEstimator
from detector.yolo_detect import ObjectDetectionYolo, VideoProcessor
from detector.visualize import BBoxVisualizer
from config import config
import torch
import cv2
import copy


class DrownDetector(object):
    def __init__(self, path=config.video_path):
        self.pose_estimator = PoseEstimator()
        self.video_processor = VideoProcessor()
        self.object_detector = ObjectDetectionYolo()
        self.BBV = BBoxVisualizer()
        self.video_path = path
        self.cap = cv2.VideoCapture(self.video_path)
        self.img = []
        self.img_black = []

    def process(self):
        cnt = 0
        while True:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, config.frame_size)
                img, orig_img, im_name, im_dim_list = self.video_processor.process(frame)
                with torch.no_grad():
                    inps, orig_img, boxes, scores, pt1, pt2 = self.object_detector.process(img, orig_img, im_name, im_dim_list)
                    cv2.imshow("bbox", self.BBV.visualize(boxes, copy.deepcopy(frame)))
                    key_points, self.img, self.img_black = self.pose_estimator.process_img(inps, orig_img, boxes, scores, pt1, pt2)
                    if len(img) > 0 and len(key_points) > 0:
                        # for key_point in key_points:

                        self.__show_img()
                    else:
                        self.__show_img()
                cnt += 1
            else:
                self.cap.release()
                cv2.destroyAllWindows()
                break

    def __show_img(self):
        cv2.imshow("result", self.img)
        cv2.moveWindow("result", 1200, 90)
        cv2.imshow("result_black", self.img_black)
        cv2.moveWindow("result_black", 1200, 540)
        cv2.waitKey(1)


if __name__ == '__main__':
    DD = DrownDetector()
    DD.process()
