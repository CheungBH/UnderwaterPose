from tracker.sort import Sort


class ObjectTracker(object):
    def __init__(self):
        self.tracker = Sort()
        self.ske_bbox = []

    def locate_skeleton(self, bboxes, skeletons):
        for skeleton in skeletons:
            pass

    def judge_in_bbox(self, coord, box):
        pass