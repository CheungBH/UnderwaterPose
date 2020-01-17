import cv2


class IDVisualizer(object):
    def __init__(self, with_bbox=True):
        self.with_bbox = with_bbox
        self.id_color = (0, 255, 0)
        self.box_color = (0, 0, 255)

    def plot(self, id2bbox, img):
        for idx in range(len(id2bbox)):
            [x1, y1, x2, y2] = list(id2bbox.values())[idx]
            cv2.putText(img, "id{}".format(list(id2bbox.keys())[idx]), (int((x1 + x2)/2), int(y1)),
                        cv2.FONT_HERSHEY_PLAIN, 2, self.id_color, 2)
            if self.with_bbox:
                img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), self.box_color, 2)
        return img
