from estimator.pose_estimator import PoseEstimator
from detector.yolo_detect import ObjectDetectionYolo
from detector.visualize import BBoxVisualizer
from tracker.track import ObjectTracker
from tracker.visualize import IDVisualizer
from config import config
import torch
import cv2
import copy


class DrownDetector(object):
    def __init__(self, path=config.video_path):
        self.pose_estimator = PoseEstimator()
        self.object_detector = ObjectDetectionYolo()
        self.BBV = BBoxVisualizer()
        self.IDV = IDVisualizer()
        self.object_tracker = ObjectTracker()
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
                with torch.no_grad():
                    inps, orig_img, boxes, scores, pt1, pt2 = self.object_detector.process(frame)
                    if boxes is not None:
                        # cv2.imshow("bbox", self.BBV.visualize(boxes, copy.deepcopy(frame)))
                        key_points, self.img, self.img_black = self.pose_estimator.process_img(inps, orig_img, boxes, scores, pt1, pt2)
                        if len(key_points) > 0:
                            id2ske, id2bbox = self.object_tracker.track(boxes, key_points)
                            cv2.imshow("id_bbox", self.IDV.plot_bbox_id(id2bbox, copy.deepcopy(frame)))
                            cv2.imshow("id_ske", self.IDV.plot_skeleton_id(id2ske, copy.deepcopy(frame)))
                            # for key_point in key_points:

                            self.__show_img()
                        else:
                            self.__show_img()
                    else:
                        # cv2.imshow("bbox", frame)
                        # cv2.imshow("id", frame)
                        self.img, self.img_black = frame, frame
                        self.__show_img()
                cnt += 1
                print(cnt)
            else:
                self.cap.release()
                cv2.destroyAllWindows()
                break

    def __show_img(self):
        # cv2.moveWindow("bbox", 600, 90)
        # cv2.moveWindow("id", 600, 540)
        cv2.imshow("result", self.img)
        cv2.moveWindow("result", 1200, 90)
        cv2.imshow("result_black", self.img_black)
        cv2.moveWindow("result_black", 1200, 540)
        cv2.waitKey(1)


if __name__ == '__main__':
    DD = DrownDetector()
    DD.process()
