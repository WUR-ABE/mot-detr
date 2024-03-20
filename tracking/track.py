import numpy as np
from filterpy.kalman import KalmanFilter
import random


class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3
    Reasoned = 4


class Track:
    """
    A single target track with state space:
        - x, y, z coordinates relative to camera
        - class: Object class (TODO: get PMF out of Mask-RCNN)

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state (x, y, z) distribution.
    covariance : ndarray
        Covariance matrix of the initial state (x, y, z) distribution.
    track_class : int
        Class of the track.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int or None
        The maximum number of consecutive misses before the track state is
        set to `Deleted`. If None, tracks are never deleted.

    Attributes
    ----------
    kf : filterpy.KalmanFilter
        Kalman filter that containes the state of the track.
    track_class : int
        Class of the track. 0-cup | 1-book | 2-fruit
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    """

    def __init__(self, mean, covariance, track_class, track_id, n_init, max_age, bbox, confidence, features):
        self.kf = KalmanFilter(dim_x=3, dim_z=3)
        self.kf.x = mean
        self.kf.P = covariance
        self.kf.F = np.eye(3, 3)
        self.kf.H = np.eye(3, 3)
        self.track_class = track_class
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative

        self._n_init = n_init
        self._max_age = max_age

        self.last_bbox = bbox
        self.track_last_conf = confidence

        self.features = []
        self.features.append(features.unsqueeze(0))

        self.alpha = 0.9
        self.smooth_feat = features

        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def predict(self, Q):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        Q : ndarray
            A 3x3 matrix which contains the process covariance.

        """
        self.kf.predict(Q=Q)
        self.age += 1
        self.time_since_update += 1

    def update(self, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        detection : Detection
            The associated detection.

        """
        alpha = self.hits / (self.hits + 1)
        alpha = max(0.9, alpha)

        self.kf.update(z=detection.z, R=detection.R)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed
        self.track_class = detection.det_class

        self.last_bbox = detection.bbox
        self.track_last_conf = detection.confidence
        self.features.append(detection.features.unsqueeze(0))
        self.smooth_feat = alpha * self.smooth_feat + (1 - alpha) * detection.features

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step)."""
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def mark_confirmed(self):
        """Mark this track as confirmed"""
        self.state = TrackState.Confirmed

    def mark_reasoned(self):
        """Mark this track as reasoned."""
        self.state = TrackState.Reasoned

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed)."""
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted

    def __repr__(self):
        return f"""Track with
        - ID {self.track_id}
        - position {self.kf.x}
        - state {self.state}
        not updated since {self.time_since_update}"""
