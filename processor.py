from estimator.detector import ObjectDetection, DetectionProcessor, VideoProcessor
from estimator.pose_estimator import PoseEstimator as PoseEstimator
from estimator.datatset import Mscoco
from estimator.opt import opt
from SPPE.src.main_fast_inference import *
from config import config
from tracker.sort import Sort
import torch

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
            try:
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
                ske_img, skeleton, ske_black_img, boxes = self.data_recorder.process(boxes, scores, hm, pt1, pt2, orig_img,
                                                                              im_name.split('/')[-1])
                return skeleton, ske_img, ske_black_img

            except:
                return [], frame, frame


class ImageProcessorWithTracking(object):
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
        self.tracker = Sort()
        self.Tensor = torch.cuda.FloatTensor
        self.ids = []
        self.bboxes = []
        self.skeletons = []
        self.id2box = {}
        self.box2skeleton = {}

    def process_img(self, frame):
        img, orig_img, im_name, im_dim_list = self.video_processor.process(frame)

        with torch.no_grad():
            inps, orig_img, im_name, boxes, scores, pt1, pt2 = self.object_detector.process(img, orig_img, im_name,
                                                                                            im_dim_list)

            inps, orig_img, im_name, boxes, scores, pt1, pt2 = self.detection_processor.process(inps, orig_img, im_name,
                                                                                                boxes, scores, pt1, pt2)

            detections = [box.tolist() + [0.999, 0.999, 0] for box in boxes]
            detection_tensor = self.Tensor(detections)
            if detection_tensor is not None:
                tracked_objects = self.tracker.update(detection_tensor.cpu())
                self.id2box = {item[5]: item[:4].tolist() for item in tracked_objects}

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
            ske_img, self.skeletons, ske_black_img, self.bboxes = self.data_recorder.process(boxes, scores, hm, pt1, pt2, orig_img,
                                                                          im_name.split('/')[-1])


            #self.box2skeleton = {boxes[idx].tolist(): skeleton[idx] for idx in range(len(boxes.tolist()))}
            # a = {}
            # for i in range(len(boxes.tolist())):
            #     a[boxes[i].tolist()] = skeleton[i]
            # print(a)
            # id2skeleton = self.reallocate_id()
            return id2skeleton, ske_img, ske_black_img

            # except:
            #     return {}, frame, frame

    def reallocate_id(self):
        return {reid: self.box2skeleton[self.id2box[reid]] for reid in self.id2box.keys()}
