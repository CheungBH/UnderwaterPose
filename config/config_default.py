import torch

device = "cuda:0"

gray_yolo_cfg = "../../Documents/Pose_demo/src/debug/model/gray/1007/13/yolov3-spp-1cls-leaky.cfg"
gray_yolo_weights = "../../Documents/Pose_demo/src/debug/model/gray/1007/13/best.weights"
black_yolo_cfg = "../../Documents/Pose_demo/src/debug/model/black/1007/2/yolov3-original-1cls-leaky.cfg"
black_yolo_weights = "../../Documents/Pose_demo/src/debug/model/black/1007/2/best.weights"
rgb_yolo_cfg = ""
rgb_yolo_weights = ""

pose_weight = "weights/sppe/duc_se.pth"
pose_cfg = None

video_path = "video/0622_1_Trim.mp4"
water_top = 40

RNN_frame_length = 4
RNN_backbone = "TCN"
RNN_class = ["stand", "drown"]
RNN_weight = "weights/RNN/TCN_struct1_2020-07-08-20-02-32.pth"
TCN_single = True

'''
----------------------------------------------------------------------------------------------------------------
'''

# For yolo
confidence = 0.05
num_classes = 80
nms_thresh = 0.33
input_size = 416

# For pose estimation
input_height = 320
input_width = 256
output_height = 80
output_width = 64
fast_inference = True
pose_batch = 80

pose_backbone = "seresnet101"
pose_cls = 17
DUCs = [480, 240]


# For detection
frame_size = (720, 540)
store_size = (frame_size[0]*4, frame_size[1]*3)
resize_ratio = 0.5
show_size = (1440, 840)

black_box_threshold = 0.7
gray_box_threshold = 0.7

libtorch = None

import os
pose_option = os.path.join("/".join(pose_weight.replace("\\", "/").split("/")[:-1]), "option.pkl")
if os.path.exists(pose_option):
    info = torch.load(pose_option)
    pose_backbone = info.backbone
    pose_cfg = info.struct
    pose_cls = info.kps
    DUC_idx = info.DUC

    output_height = info.outputResH
    output_width = info.outputResW
    input_height = info.inputResH
    input_width = info.inputResW