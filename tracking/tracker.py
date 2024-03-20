import numpy as np
import cv2

from . import cost
from .track import Track


class Tracker:
    def __init__(self, max_age=None, n_init=5, id_start=0, w=0.5, gating_min=0.4, gating_max=0.8):
        self.max_age = max_age
        self.n_init = n_init
        self.tracks = []
        self.deleted_tracks = []
        self._next_id = 3 + id_start
        assert w >= 0 and w <= 1
        self.w = w
        self.gating_min = gating_min
        self.gating_max = gating_max

    def predict(self, Q):
        for track in self.tracks:
            track.predict(Q=Q)

    def update(self, detections):
        matches, unmatched_tracks, unmatched_detections = self.match(detections)

        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            if self.n_init == 0:
                self.start_confirmed_track(detections[detection_idx])
            else:
                self.start_track(detections[detection_idx])
        self.deleted_tracks = [t for t in self.tracks if t.is_deleted()]
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

    def match(self, detections):
        track_indices = np.arange(len(self.tracks))
        detection_indices = np.arange(len(detections))

        if len(detection_indices) == 0 or len(track_indices) == 0:
            return [], track_indices, detection_indices  # Nothing to match.

        det_positions = []
        det_classes = []
        det_features = []
        for det in detections:
            det_positions.append(det.z)
            det_classes.append(det.det_class)
            det_features.append(det.features.unsqueeze(0))

        track_positions = []
        track_covariances = []
        track_classes = []
        track_features = []

        for track in self.tracks:
            track_positions.append(track.kf.x)
            track_covariances.append(track.kf.P)
            track_classes.append(track.track_class)
            # track_features.append(track.features[-1])
            track_features.append(track.smooth_feat.unsqueeze(0))

        mahalanobis_cost_matrix = cost.mahalanobis_cost_matrix(
            row_positions=track_positions,
            row_covariances=track_covariances,
            col_positions=det_positions,
        )

        feat_cost_matrix = cost.cosine_cost_matrix(row_feats=track_features, col_feats=det_features)
        feat_cost_matrix[feat_cost_matrix >= self.gating_max] += 1e5
        mahalanobis_cost_matrix = mahalanobis_cost_matrix / mahalanobis_cost_matrix.max()
        # feat_cost_matrix = feat_cost_matrix / feat_cost_matrix.max()

        cost_matrix = (1 - self.w) * mahalanobis_cost_matrix + self.w * feat_cost_matrix
        # Associate confirmed tracks using appearance features.
        (
            matches,
            unmatched_tracks,
            unmatched_detections,
        ) = cost.min_cost_matching(
            cost_matrix,
            1e4,
            track_indices,
            detection_indices,
        )
        return matches, unmatched_tracks, unmatched_detections

    def start_track(self, detection):
        self.tracks.append(
            Track(
                detection.z,
                detection.R,
                detection.det_class,
                self._next_id,
                self.n_init,
                self.max_age,
                detection.bbox,
                detection.confidence,
                detection.features,
            )
        )
        self._next_id += 1

    def start_confirmed_track(self, detection):
        track = Track(
            detection.z,
            detection.R,
            detection.det_class,
            self._next_id,
            self.n_init,
            self.max_age,
            detection.bbox,
            detection.confidence,
            detection.features,
        )
        track.mark_confirmed()

        self.tracks.append(track)
        self._next_id += 1

        return track.track_id

    def get_tracks_in_img(self, img):
        for t in self.tracks:
            if t.time_since_update == 0:
                bbox = t.last_bbox.astype(np.int32)
                cv2.putText(
                    img,
                    str(t.track_id),
                    (int((bbox[2] + bbox[0]) / 2), int((bbox[3] + bbox[1]) / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    t.color,
                    thickness=2,
                )
                img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), t.color, 2)
        return img
