from estimator.pose_estimator import PoseEstimator
from detector.yolo_detect import ObjectDetectionYolo, VideoProcessor
from config import config
import torch
import cv2


class DrownDetector(object):
    def __init__(self, path=config.video_path):
        self.pose_estimator = PoseEstimator()
        self.video_processor = VideoProcessor()
        self.object_detector = ObjectDetectionYolo()
        self.video_path = path
        self.cap = cv2.VideoCapture(self.video_path)

    def process(self):
        cnt = 0
        while True:
            ret, frame = self.cap.read()
            frame = cv2.resize(frame, (540, 360))
            if ret:
                img, orig_img, im_name, im_dim_list = self.video_processor.process(frame)
                with torch.no_grad():
                    inps, orig_img, boxes, scores, pt1, pt2 = self.object_detector.process(img, orig_img,
                                                                                                    im_name,
                                                                                                    im_dim_list)
                    key_point, img, img_black = self.pose_estimator.process_img(inps, orig_img, boxes, scores, pt1, pt2)
                    if len(img) > 0 and len(key_point) > 0:


                        # pass


                        cv2.imshow("result", img)
                        cv2.imshow("result_black", img_black)
                        cv2.waitKey(1)
                    else:
                        cv2.imshow("result", img)
                        cv2.imshow("result_black", img_black)
                        cv2.waitKey(1)
                cnt += 1
            else:
                self.cap.release()
                cv2.destroyAllWindows()
                break


if __name__ == '__main__':
    DD = DrownDetector()
    DD.process()