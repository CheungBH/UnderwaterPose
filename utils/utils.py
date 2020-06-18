import numpy as np
import math
import cv2
import torch

image_normalize_mean = [0.485, 0.456, 0.406]
image_normalize_std = [0.229, 0.224, 0.225]


class Utils(object):
    def __init__(self):
        pass

    @staticmethod
    def get_angle(center_coor, coor2, coor3):
        L1 = Utils.cal_dis(coor2,coor3)
        L2 = Utils.cal_dis(center_coor,coor3)
        L3 = Utils.cal_dis(center_coor,coor2)
        Angle = Utils.cal_angle(L1,L2,L3)
        return Angle

    @staticmethod
    def cal_dis(coor1, coor2):
        out = np.square(coor1[0] - coor2[0]) + np.square(coor1[1] - coor2[1])
        return np.sqrt(out)

    @staticmethod
    def cal_angle(L1, L2, L3):
        out = (np.square(L2) + np.square(L3) - np.square(L1)) / (2 * L2 * L3)
        try:
            return math.acos(out) * (180 / math.pi)
        except ValueError:
            return 180

    @staticmethod
    def image_normalize(image, size=224):
        image_array = cv2.resize(image, (size, size))
        image_array = np.ascontiguousarray(image_array[..., ::-1], dtype=np.float32)
        image_array = image_array.transpose((2, 0, 1))
        for channel, _ in enumerate(image_array):
            image_array[channel] /= 255.0
            image_array[channel] -= image_normalize_mean[channel]
            image_array[channel] /= image_normalize_std[channel]
        image_tensor = torch.from_numpy(image_array).float()
        return image_tensor


def cut_image(img, bottom=0, top=0, left=0, right=0):
    height, width = img.shape[0], img.shape[1]
    return np.asarray(img[top: height - bottom, left: width - right])


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def gray3D(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray", gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

if __name__ == '__main__':
    ut = Utils()
    # res = ut.time_to_string("10.0000")
    # print(res)
    res = ut.get_angle([0, 0], [1, -1], [0, 1])
    print(res)
