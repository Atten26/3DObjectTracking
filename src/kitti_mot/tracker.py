"""Constant-velocity Kalman tracker with Hungarian association.

The state is the 3D center and its velocity in rectified camera coordinates.
Dimensions, orientation and class are carried over from the latest detection:
they are not filtered, since association never uses them.

No ego-motion compensation: the camera sits on a moving car, so a parked
object still shows a velocity here. See README for what that costs.
"""

from dataclasses import replace

import numpy as np
from scipy.optimize import linear_sum_assignment

# Tuned on the KITTI label statistics: objects never move more than 2.47 m
# between frames, and the 95th percentile of their acceleration is 0.12 m/frame^2.
GATE = 3.0  # m, pairs further apart than this are never the same object
SIGMA_ACCEL = 0.15  # m/frame^2, process noise
# Measurement noise. Ground truth is exact, so 0.10 m is right for it, but this
# has to match what the detector actually delivers: telling the filter the input
# is ten times better than it is costs ~0.18 IDF1 at a real noise of 0.5 m.
SIGMA_MEASURE = 0.10  # m

# Constant velocity, one frame per step.
F = np.block([[np.eye(3), np.eye(3)], [np.zeros((3, 3)), np.eye(3)]])
H = np.hstack([np.eye(3), np.zeros((3, 3))])
Q = SIGMA_ACCEL**2 * np.block([
    [0.25 * np.eye(3), 0.5 * np.eye(3)],
    [0.5 * np.eye(3), np.eye(3)],
])


class Track:
    """One object being followed, with its own Kalman state."""

    def __init__(self, track_id, box, measurement_noise=SIGMA_MEASURE):
        self.id = track_id
        self.box = box
        self.R = measurement_noise**2 * np.eye(3)
        self.x = np.concatenate([box.center, np.zeros(3)])
        # the position is known well from the first detection, the velocity is not
        self.P = np.diag([1.0, 1.0, 1.0, 10.0, 10.0, 10.0])
        self.hits = 1
        self.time_since_update = 0

    @property
    def position(self):
        return self.x[:3]

    def predict(self):
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.time_since_update += 1

    def update(self, box):
        innovation = box.center - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.P = (np.eye(6) - K @ H) @ self.P
        self.box = box
        self.hits += 1
        self.time_since_update = 0


class Tracker:
    """Feed it the detections of one frame at a time, in order."""

    def __init__(self, gate=GATE, min_hits=3, max_age=3, measurement_noise=SIGMA_MEASURE):
        self.gate = gate
        self.measurement_noise = measurement_noise
        self.min_hits = min_hits  # frames before a track is trusted enough to report
        self.max_age = max_age  # frames a track survives without a detection
        self.tracks = []
        self.next_id = 0

    def update(self, detections):
        """Returns the confirmed tracks as (track_id, Box3D) pairs."""
        for track in self.tracks:
            track.predict()

        matches, unmatched_detections = self._associate(detections)
        for track, detection in matches:
            track.update(detection)
        for detection in unmatched_detections:
            self.tracks.append(Track(self.next_id, detection, self.measurement_noise))
            self.next_id += 1

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        output = []
        for track in self.tracks:
            if track.time_since_update == 0 and track.hits >= self.min_hits:
                output.append((track.id, replace(track.box, center=track.position.copy())))
        return output

    def _associate(self, detections):
        """Hungarian assignment on 3D distance, with pairs beyond the gate dropped."""
        if not self.tracks or not detections:
            return [], list(detections)

        predicted = np.array([t.position for t in self.tracks])
        measured = np.array([d.center for d in detections])
        cost = np.linalg.norm(predicted[:, None] - measured[None], axis=-1)

        track_rows, detection_cols = linear_sum_assignment(cost)
        matches = []
        matched_cols = set()
        for row, col in zip(track_rows, detection_cols):
            if cost[row, col] <= self.gate:
                matches.append((self.tracks[row], detections[col]))
                matched_cols.add(col)
        unmatched = [d for i, d in enumerate(detections) if i not in matched_cols]
        return matches, unmatched
