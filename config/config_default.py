device = "cuda:0"

# For yolo
confidence = 0.05
num_classes = 80
nms_thresh = 0.33
input_size = 416

black_yolo_cfg = "yolo/cfg/yolov3-spp-1cls.cfg"
black_yolo_weights = 'weights/yolo/black/freeze.weights'
gray_yolo_cfg = "yolo/cfg/prune_0.93_keep_0.1.cfg"
gray_yolo_weights = 'weights/yolo/gray/best.weights'

# For pose estimation
input_height = 320
input_width = 256
output_height = 80
output_width = 64
fast_inference = True
pose_batch = 80

pose_backbone = "seresnet101"
pose_weight = "weights/sppe/duc_se.pth"
pose_cfg = None
pose_cls = 17

DUCs = [480, 240]


track_idx = "all"    # If all idx, track_idx = "all"
track_plot_id = ["all"]   # If all idx, track_plot_id = ["all"]
assert track_idx == "all" or isinstance(track_idx, int)

# For detection
video_path = "Video/45_Trim2.mp4"
frame_size = (540, 360)

plot_bbox = True
plot_kps = True
plot_id = True
