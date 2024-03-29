import torch
from src.utils.utils import cal_center_point

max_box_store = 50
box_ratio_thresh = 2.8
cal_hw_num = 10
hw_percent_ratio = 0.8


class Box:
    def __init__(self, box):
        self.boxes = box.unsqueeze(dim=0)
        self.centers = [cal_center_point(box)]
        self.ratios = torch.FloatTensor([])
        self.curr_box = box

    def append(self, box):
        self.curr_box = box
        self.boxes = torch.cat([self.boxes, box.unsqueeze(dim=0)], dim=0)
        self.centers.append(cal_center_point(box))
        if len(self) > max_box_store:
            self.boxes = self.boxes[1:]
            self.centers = self.centers[1:]

    def __len__(self):
        return len(self.boxes)

    def cal_size_ratio(self):
        tmp = self.boxes[-cal_hw_num:] if len(self) > cal_hw_num else self.boxes
        self.ratios = (tmp[:,3]-tmp[:,1])/(tmp[:,2]-tmp[:,0])
        # print(self.ratios)

    def get_size_ratio_info(self):
        return True if sum((self.ratios > box_ratio_thresh).float())/len(self.ratios) >= hw_percent_ratio else False

    def cal_curr_hw(self):
        h, w = self.curr_box[3] - self.curr_box[1], self.curr_box[2] - self.curr_box[0]
        return h, w

    def text_color(self, r):
        return "yellow" if r > box_ratio_thresh else "purple"

    def curr_center(self):
        return cal_center_point(self.curr_box)

    def curr_top(self):
        x1, y1, x2, y2 = self.curr_box
        return int(x1), int(y1)-20

    def curr_box_location(self):
        x1, y1, x2, y2 = self.curr_box
        return (int(x1), int(y1)), (int(x2), int(y2))
