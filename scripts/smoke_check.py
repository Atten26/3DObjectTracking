"""Smoke check: the projected 3D box should cover the 2D bbox from the same label row."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kitti_mot.dataset import load_labels, load_p2
from kitti_mot.viz import project_box

TRAINING = ROOT / "data" / "kitti_tracking" / "training"


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


for sequence in ("0000", "0011"):
    boxes = load_labels(TRAINING / "label_02" / f"{sequence}.txt")
    p2 = load_p2(TRAINING / "calib" / f"{sequence}.txt")
    assert p2.shape == (3, 4)
    assert all(box.type != "DontCare" and box.track_id >= 0 for box in boxes)

    ious = []
    behind = 0
    for box in boxes:
        points = project_box(box, p2)
        if points is None:
            behind += 1
            continue
        if box.truncated > 0 or box.occluded > 1:
            continue  # the annotated 2D bbox is clipped by the border or unreliable
        ious.append(iou(
            (points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()),
            box.bbox,
        ))

    median = float(np.median(ious))
    print(
        f"{sequence}: {len(boxes)} boxes, {len(set(b.track_id for b in boxes))} tracks, "
        f"{len(set(b.type for b in boxes))} classes, {behind} behind the camera, "
        f"median 2D/3D IoU {median:.3f} over {len(ious)} boxes"
    )
    assert median > 0.8, f"projection looks wrong on sequence {sequence}"

print("ok")
