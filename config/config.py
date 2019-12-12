import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cuda_flag = torch.cuda.is_available()

X_vector_dim = 36
time_step = 15
y_vector_dim = 5
LSTM_classes = ["null", "FollowThrough", "Standing", "BackSwing", "Downswing"]
LSTM_model = ''
pose_model = ''

input_size_dict = {"inception": 299, "resnet18": 224, "resnet34": 224, "resnet50": 224, "resnet101": 224,
                   "resnet152": 224, "CNN": 224, "LeNet": 28, "mobilenet":224, "shufflenet": 224}

golf_video_path = 'Video/golf/2people.mp4'
golf_webcam_num = 0
CNN_golf_model = 'models/CNN/golf/golf_ske_shufflenet_2019-10-14-08-36-18.pth'
CNN_golf_pre_train_model = 'shufflenet'
CNN_golf_classes = ["Backswing", "FollowThrough", "Standing"]
golf_image_input_size = input_size_dict[CNN_golf_pre_train_model]

golf_static_step = 10


yoga_video_path = 'Video/yogatest/00_Trim.mp4'
yoga_webcam_num = 1
CNN_yoga_model = 'models/CNN/yoga/yoga_1104.pth'
CNN_yoga_pre_train_model = 'shufflenet'
# CNN_yoga_class = ["bow", "catnew"]
CNN_yoga_class = ["boat", "boat", "chair", "chair", "tree", "tree", "triangle", "triangle"]
yoga_image_input_size = input_size_dict[CNN_yoga_pre_train_model]

push_up_video_path = "Video/push_up/wrong2_Trim.mp4"

sit_up_video_path = "Video/sit_up/sit_up1_Trim.mp4"

squat_up_side_video_path = "Video/golf/doing/00.avi"

squat_up_front_video_path = "Video/golf/doing/00.avi"
