"""A detector stand-in, built out of the ground truth boxes.

Ground truth tracks are continuous by construction - neither of our two sequences
has a single gap - so feeding them straight to the tracker never exercises the
track lifecycle. This injects the three failure modes a real detector has:
missed objects, spurious objects, and noisy positions.

What matters here is that failures can be made to *persist*. A real detector does
not lose an object for one frame and find it again; it loses it for as long as the
truck in front stays in front. It does not hallucinate a pedestrian once; it
hallucinates the same bush in every frame of the sequence. Independent noise is
the easy case, and `occlusion_frames=1, ghost_frames=1` reproduces it.

Detections always carry track_id -1: the tracker must never see the answer.
"""

from dataclasses import replace

import numpy as np


class Detector:
    """Stateful, so call it once per frame, in order.

    miss_rate        fraction of the time an object is missing, however it is spread
    occlusion_frames mean length of one disappearance (1 = independent misses)
    false_positives  mean number of spurious objects present per frame
    ghost_frames     mean lifetime of one spurious object (1 = independent)
    position_noise   standard deviation, in meters, on each coordinate

    The two rates are expressed as totals, not as per-frame events, so raising
    the durations spreads the same amount of damage differently instead of
    adding more of it.
    """

    def __init__(self, rng, miss_rate=0.0, false_positives=0.0, position_noise=0.0,
                 occlusion_frames=1.0, ghost_frames=1.0):
        self.rng = rng
        self.position_noise = position_noise
        self.occlusion_frames = max(occlusion_frames, 1.0)
        self.ghost_frames = max(ghost_frames, 1.0)
        # A longer failure has to fire proportionally less often, so that the
        # amount of damage stays put and only its shape changes - otherwise a
        # comparison between independent and persistent noise is really just a
        # comparison between less noise and more of it.
        #
        # An object alternates visible and hidden. Each visible frame triggers a
        # hiding with probability p, and the triggering frame is itself hidden, so
        # one cycle is (1/p - 1) visible frames and L hidden ones. Setting
        # L*p / (1 - p + L*p) = miss_rate gives the p below. At L = 1 it collapses
        # to p = miss_rate, which is the independent case.
        L, m = self.occlusion_frames, miss_rate
        self.occlusion_start = m / (L * (1.0 - m) + m) if m else 0.0
        self.ghost_start = false_positives / self.ghost_frames
        self.hidden = {}  # track_id -> frames still to stay hidden
        self.ghosts = []  # [[center, frames still to live, source box]]

    def __call__(self, boxes):
        detections = []
        for box in boxes:
            if self._is_hidden(box.track_id):
                continue
            detections.append(replace(box, center=self._jitter(box.center), track_id=-1))
        detections.extend(self._spurious(boxes))
        return detections

    def _is_hidden(self, track_id):
        remaining = self.hidden.get(track_id, 0)
        if remaining > 0:
            self.hidden[track_id] = remaining - 1
            return True
        if self.rng.random() < self.occlusion_start:
            self.hidden[track_id] = self._duration(self.occlusion_frames) - 1
            return True
        return False

    def _spurious(self, boxes):
        """Ghosts persist in place, so a long-lived one will get past min_hits."""
        surviving, detections = [], []
        for center, remaining, source in self.ghosts:
            if remaining <= 0:
                continue
            detections.append(replace(source, center=self._jitter(center), track_id=-1))
            surviving.append([center, remaining - 1, source])
        self.ghosts = surviving

        if boxes:
            for _ in range(self.rng.poisson(self.ghost_start)):
                source = boxes[self.rng.integers(len(boxes))]
                direction = self.rng.normal(0.0, 1.0, 3)
                direction[1] = 0.0  # keep it on the ground plane
                direction /= np.linalg.norm(direction)
                center = source.center + direction * self.rng.uniform(5.0, 15.0)
                self.ghosts.append([center, self._duration(self.ghost_frames), source])
        return detections

    def _duration(self, mean):
        return int(self.rng.geometric(1.0 / mean)) if mean > 1.0 else 1

    def _jitter(self, center):
        if not self.position_noise:
            return center
        return center + self.rng.normal(0.0, self.position_noise, 3)
