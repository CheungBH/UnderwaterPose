from estimator.opt import opt as args
import cv2
from processor import ImageProcessor as ImageProcessor

IP = ImageProcessor()
nece_point = [0, 11, 12]


def judge_slope(body_ls):
    hip_middle = [(body_ls[1][0] + body_ls[2][0])/2, (body_ls[1][1] + body_ls[2][1])/2]
    slope = (hip_middle[0] - body_ls[0][0])/(hip_middle[1] - body_ls[0][1])
    if abs(slope) < 1:
        return "vertical"
    else:
        return "horizontal"


def run(cam_num):
    frm_cnt = 0
    cap = cv2.VideoCapture(cam_num)

    while True:
        frm_cnt += 1
        ret, frame = cap.read()
        frame = cv2.resize(frame, (540, 360))

        if ret:
            key_point, img_fast, img = IP.process_img(frame)
            if len(img) > 0 and len(key_point) > 0:
                # coord = [key_point[idx] for idx in nece_point]
                # state = judge_slope(coord)
                # print("The swimmer's body is {}".format(state))
                # if state == "vertical":
                #     pass
                #     # vertical detection
                # else:
                #     pass
                #     # horizontal detection
                #
                cv2.imshow("result_fast", img_fast)
                cv2.imshow("result", img)
                cv2.waitKey(1)
            else:
                cv2.imshow("result_fast", img_fast)
                cv2.imshow("result", img)
                cv2.waitKey(1)
        else:
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    video_path = "Video/multiple.mp4"
    run(video_path)