from estimator.detector import ObjectDetection, DetectionProcessor, VideoProcessor
from estimator.pose_estimator import PoseEstimator
from estimator.datatset import Mscoco
from estimator.opt import opt
from SPPE.src.main_fast_inference import *
from config import config

args = opt
args.dataset = 'coco'

step = config.golf_static_step
batchSize = args.posebatch


class ImageProcessor(object):
    def __init__(self):
        self.object_detector = ObjectDetection()
        self.detection_processor = DetectionProcessor()
        self.data_recorder = PoseEstimator()
        self.video_processor = VideoProcessor()

        pose_dataset = Mscoco()
        if args.fast_inference:
            self.pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
        else:
            self.pose_model = InferenNet(4 * 1 + 1, pose_dataset)
        self.pose_model.cuda()
        self.pose_model.eval()

    def process_img(self, frame):
        img, orig_img, im_name, im_dim_list = self.video_processor.process(frame)

        with torch.no_grad():
            inps, orig_img, im_name, boxes, scores, pt1, pt2 = self.object_detector.process(img, orig_img, im_name,
                                                                                            im_dim_list)
            inps, orig_img, im_name, boxes, scores, pt1, pt2 = self.detection_processor.process(inps, orig_img, im_name,
                                                                                                boxes, scores, pt1, pt2)
            # try:
            datalen = inps.size(0)
            leftover = 0
            if (datalen) % batchSize:
                leftover = 1
            num_batches = datalen // batchSize + leftover
            hm = []

            for j in range(num_batches):
                inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)].cuda()
                hm_j = self.pose_model(inps_j)
                hm.append(hm_j)
            hm = torch.cat(hm).cpu().data
            ske_img, skeleton, ske_black_img = self.data_recorder.process(boxes, scores, hm, pt1, pt2, orig_img,
                                                                          im_name.split('/')[-1])
            return skeleton, ske_img, ske_black_img

            # except:
            #     return [], frame, frame

