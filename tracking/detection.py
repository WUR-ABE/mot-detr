class Detection(object):
    def __init__(self, position, R, det_class, bbox, confidence, features):
        self.z = position
        self.R = R
        self.det_class = det_class
        self.bbox = bbox
        self.confidence = confidence
        self.features = features
