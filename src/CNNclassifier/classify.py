from config.config import CNN_class
import cv2
from src.utils.utils import cut_image_box
from .inference import CNNInference


class CNNClassifier:
    def __init__(self):
        self.CNN_model = CNNInference()
        self.pred = {}

    def classify(self, img):
        out = self.CNN_model.predict(img)
        idx = out[0].tolist().index(max(out[0].tolist()))
        pred = CNN_class[idx]
        print("The prediction is {}".format(pred))
        return pred

    def classify_all(self, im, id2bbox):
        self.pred = {}
        for idx, box in id2bbox.items():
            cut_img = cut_image_box(im, box)
            pred = self.classify(cut_img)
            self.pred[idx] = pred
            text_location = (int((box[0] + box[2]) / 2)), int((box[1]) + 50)
            cv2.putText(im, CNN_class[idx], text_location, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
        return im, self.pred
