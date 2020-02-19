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


def crop_from_dets(img, boxes, inps, pt1, pt2):
    '''
    Crop human from origin image according to Dectecion Results
    '''

    imght = img.size(1)
    imgwidth = img.size(2)
    tmp_img = img
    tmp_img[0].add_(-0.406)
    tmp_img[1].add_(-0.457)
    tmp_img[2].add_(-0.480)
    for i, box in enumerate(boxes):
        upLeft = torch.Tensor(
            (float(box[0]), float(box[1])))
        bottomRight = torch.Tensor(
            (float(box[2]), float(box[3])))

        ht = bottomRight[1] - upLeft[1]
        width = bottomRight[0] - upLeft[0]

        scaleRate = 0.3

        upLeft[0] = max(0, upLeft[0] - width * scaleRate / 2)
        upLeft[1] = max(0, upLeft[1] - ht * scaleRate / 2)
        bottomRight[0] = max(
            min(imgwidth - 1, bottomRight[0] + width * scaleRate / 2), upLeft[0] + 5)
        bottomRight[1] = max(
            min(imght - 1, bottomRight[1] + ht * scaleRate / 2), upLeft[1] + 5)

        try:
            inps[i] = cropBox(tmp_img.clone(), upLeft, bottomRight, 320, 256)
        except IndexError:
            print(tmp_img.shape)
            print(upLeft)
            print(bottomRight)
            print('===')
        pt1[i] = upLeft
        pt2[i] = bottomRight
    return inps, pt1, pt2


def cropBox(img, ul, br, resH, resW):
    ul = ul.int()
    br = (br - 1).int()
    # br = br.int()
    lenH = max((br[1] - ul[1]).item(), (br[0] - ul[0]).item() * resH / resW)
    lenW = lenH * resW / resH
    if img.dim() == 2:
        img = img[np.newaxis, :]

    box_shape = [(br[1] - ul[1]).item(), (br[0] - ul[0]).item()]
    pad_size = [(lenH - box_shape[0]) // 2, (lenW - box_shape[1]) // 2]
    # Padding Zeros
    if ul[1] > 0:
        img[:, :ul[1], :] = 0
    if ul[0] > 0:
        img[:, :, :ul[0]] = 0
    if br[1] < img.shape[1] - 1:
        img[:, br[1] + 1:, :] = 0
    if br[0] < img.shape[2] - 1:
        img[:, :, br[0] + 1:] = 0

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    src[0, :] = np.array(
        [ul[0] - pad_size[1], ul[1] - pad_size[0]], np.float32)
    src[1, :] = np.array(
        [br[0] + pad_size[1], br[1] + pad_size[0]], np.float32)
    dst[0, :] = 0
    dst[1, :] = np.array([resW - 1, resH - 1], np.float32)

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    dst_img = cv2.warpAffine(torch_to_im(img), trans,
                             (resW, resH), flags=cv2.INTER_LINEAR)

    return im_to_torch(torch.Tensor(dst_img))


if __name__ == '__main__':
    ut = Utils()
    # res = ut.time_to_string("10.0000")
    # print(res)
    res = ut.get_angle([0, 0], [1, -1], [0, 1])
    print(res)
