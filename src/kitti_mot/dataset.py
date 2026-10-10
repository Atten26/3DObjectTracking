"""Parsing of labels and calibration from the KITTI tracking benchmark."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Box3D:
    """One row of label_02/XXXX.txt (DontCare rows never reach this)."""

    frame: int
    track_id: int
    type: str
    truncated: float
    occluded: int
    alpha: float
    bbox: tuple  # (left, top, right, bottom) in pixels
    h: float
    w: float
    l: float
    center: np.ndarray  # (x, y, z) center of the BOTTOM face, rectified camera coords
    rotation_y: float


def load_labels(path):
    """Read label_02/XXXX.txt and return the list of Box3D, DontCare rows dropped."""
    boxes = []
    for line in Path(path).read_text().splitlines():
        f = line.split()
        if not f or f[2] == "DontCare":
            continue
        boxes.append(
            Box3D(
                frame=int(f[0]),
                track_id=int(f[1]),
                type=f[2],
                truncated=float(f[3]),
                occluded=int(f[4]),
                alpha=float(f[5]),
                bbox=(float(f[6]), float(f[7]), float(f[8]), float(f[9])),
                h=float(f[10]),
                w=float(f[11]),
                l=float(f[12]),
                center=np.array([float(f[13]), float(f[14]), float(f[15])]),
                rotation_y=float(f[16]),
            )
        )
    return boxes


def group_by_frame(boxes):
    """Map frame -> list of Box3D."""
    by_frame = {}
    for box in boxes:
        by_frame.setdefault(box.frame, []).append(box)
    return by_frame


def load_p2(path):
    """Read calib/XXXX.txt and return the P2 matrix (3x4).

    The keys here are P0 P1 P2 P3 R_rect Tr_velo_cam Tr_imu_velo: they are not
    the ones used by the object detection devkit.
    """
    for line in Path(path).read_text().splitlines():
        key, _, values = line.partition(":")
        if key.strip() == "P2":
            return np.array([float(v) for v in values.split()]).reshape(3, 4)
    raise ValueError(f"P2 not found in {path}")
