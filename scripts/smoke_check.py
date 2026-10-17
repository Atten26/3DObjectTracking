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


# --- metrics ---------------------------------------------------------------
# Ground truth scored against itself must come out perfect; that only rules out
# gross mistakes though, so the two cases below check that the things we actually
# count - identity switches and misses - are counted when they are really there.

from dataclasses import replace  # noqa: E402

from kitti_mot.dataset import group_by_frame  # noqa: E402
from kitti_mot.metrics import evaluate  # noqa: E402

boxes = load_labels(TRAINING / "label_02" / "0000.txt")
truth = {f: [(b.track_id, b) for b in bs] for f, bs in group_by_frame(boxes).items()}

perfect = evaluate(truth, truth)
assert perfect["MOTA"] == 1.0 and perfect["IDF1"] == 1.0 and perfect["ID switches"] == 0

# two objects that both span the whole sequence, with their identities swapped halfway
pair = {f: [(i, b) for i, b in v if i in (0, 1)] for f, v in truth.items()}
pair = {f: v for f, v in pair.items() if len(v) == 2}
midpoint = sorted(pair)[len(pair) // 2]
swapped = {
    f: [(1 - i if f >= midpoint else i, b) for i, b in v] for f, v in pair.items()
}
result = evaluate(pair, swapped)
assert result["ID switches"] == 2, f"a deliberate swap went uncounted: {result}"
assert abs(result["IDF1"] - 0.5) < 0.05, f"half of each identity is wrong, IDF1 says {result['IDF1']}"
assert result["MOTA"] > 0.99, "a swap must not be charged as a miss or a false positive"

# dropping half the frames has to show up as misses, and nothing else
half = {f: v for f, v in truth.items() if f % 2 == 0}
result = evaluate(truth, half)
expected = sum(len(v) for f, v in truth.items() if f % 2)
assert result["misses"] == expected, f"{result['misses']} misses, expected {expected}"
assert result["false positives"] == 0 and result["ID switches"] == 0

print("metrics ok")
