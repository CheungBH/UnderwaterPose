from estimator.opt import opt as args
import cv2
from processor import ImageProcessor

IP = ImageProcessor()


def run(cam_num):
    frm_cnt = 0
    cap = cv2.VideoCapture(cam_num)

    while True:
        frm_cnt += 1
        ret, frame = cap.read()

        if ret:
            key_point, img, _ = IP.process_img(frame)
            if len(img) > 0 and len(key_point) > 0:
                cv2.imshow("result", img)
                cv2.waitKey(1)
            else:
                pass
        else:
            break


cam = args.webcam
run(cam)