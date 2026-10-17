"""Turn ground truth boxes into detector-like observations.

Ground truth tracks are continuous by construction - neither of our two sequences
has a single gap - so feeding them straight to the tracker never exercises the
track lifecycle. This injects the three failure modes a real detector has:
missed objects, spurious objects, and noisy positions.

Detections always carry track_id -1: the tracker must never see the answer.
"""

from dataclasses import replace

import numpy as np


def simulate(boxes, rng, miss_rate=0.0, false_positives=0.0, position_noise=0.0):
    """Detections for one frame, from that frame's ground truth boxes.

    miss_rate        probability of dropping each real object
    false_positives  mean number of spurious objects per frame (Poisson)
    position_noise   standard deviation, in meters, added to each coordinate
    """
    detections = []
    for box in boxes:
        if rng.random() < miss_rate:
            continue
        center = box.center + rng.normal(0.0, position_noise, 3) if position_noise else box.center
        detections.append(replace(box, center=center, track_id=-1))

    # a spurious object is a real one displaced far enough to be a distinct thing
    if boxes:
        for _ in range(rng.poisson(false_positives)):
            source = boxes[rng.integers(len(boxes))]
            direction = rng.normal(0.0, 1.0, 3)
            direction[1] = 0.0  # keep it on the ground plane
            direction /= np.linalg.norm(direction)
            offset = rng.uniform(5.0, 15.0)
            detections.append(replace(source, center=source.center + direction * offset, track_id=-1))

    return detections
