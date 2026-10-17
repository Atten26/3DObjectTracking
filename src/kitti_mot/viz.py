"""Projection of 3D boxes through P2 and drawing on the images."""

import cv2
import numpy as np

# 0-3 bottom face, 4-7 top face, then the vertical pillars.
EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def box_corners(box):
    """The 8 corners of the box in rectified camera coords, as an 8x3 array.

    The box is built in its local frame (X along the length, Z along the width,
    Y pointing down), rotated around Y by rotation_y and translated onto the
    center of the bottom face.
    """
    l, w, h = box.l, box.w, box.h
    local = np.array([
        [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
        [0, 0, 0, 0, -h, -h, -h, -h],
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
    ])
    c, s = np.cos(box.rotation_y), np.sin(box.rotation_y)
    rot_y = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return (rot_y @ local).T + box.center


def project_box(box, p2):
    """Corners projected to pixels (8x2), or None if the box is behind the camera.

    The boxes already live in rectified camera coords, so P2 @ [X, Y, Z, 1] is enough.
    """
    corners = box_corners(box)
    if np.any(corners[:, 2] <= 0):
        return None
    projected = np.hstack([corners, np.ones((8, 1))]) @ p2.T
    return projected[:, :2] / projected[:, 2:3]


def track_color(track_id):
    """BGR color that is stable over time: it depends only on the track_id."""
    hue = int(track_id * 0.6180339887 * 180) % 180
    bgr = cv2.cvtColor(np.uint8([[[hue, 210, 255]]]), cv2.COLOR_HSV2BGR)
    return tuple(int(v) for v in bgr[0, 0])


def draw_box(image, points, color, label=None):
    """Draw the box wireframe and, when given, a label above it."""
    points = np.round(points).astype(np.int32)
    for i, j in EDGES:
        cv2.line(image, tuple(points[i]), tuple(points[j]), color, 2, cv2.LINE_AA)
    if label is not None:
        x = int(points[:, 0].min())
        y = int(points[:, 1].min()) - 6
        cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


def draw_frame(image, tracked_boxes, p2):
    """Draw one frame's boxes, given as (track_id, Box3D) pairs. Returns how many were skipped."""
    skipped = 0
    for track_id, box in tracked_boxes:
        points = project_box(box, p2)
        if points is None:
            skipped += 1
            continue
        draw_box(image, points, track_color(track_id), f"{box.type[:3]} {track_id}")
    return skipped


def write_video(image_paths, boxes_by_frame, p2, out_path, fps=10.0):
    """Burn the boxes onto the images and encode them. Returns (drawn, skipped)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # the codec wants even dimensions, otherwise it silently drops a row
    height, width = (n - n % 2 for n in cv2.imread(str(image_paths[0])).shape[:2])
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    drawn = skipped = 0
    for path in image_paths:
        image = cv2.imread(str(path))[:height, :width]
        if image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height))
        boxes = boxes_by_frame.get(int(path.stem), [])
        skipped += draw_frame(image, boxes, p2)
        drawn += len(boxes)
        writer.write(image)
    writer.release()
    return drawn - skipped, skipped
