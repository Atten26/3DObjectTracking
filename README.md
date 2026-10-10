# 3D Multi-Object Tracking on KITTI

Step 1: download a subset of the KITTI tracking benchmark and project the ground
truth 3D boxes onto the images.

## Setup

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/python scripts/download_subset.py
```

The download pulls sequences 0000 and 0011 of the training split into
`data/kitti_tracking/` (~425 MB). Images are extracted from the remote zip with range
requests, so the full 15 GB archive is never downloaded.

## Usage

```bash
venv/bin/python scripts/render_video.py --sequence 0000   # -> out/kitti_0000_gt.mp4
venv/bin/python scripts/smoke_check.py
```

The smoke check compares the axis-aligned bounding box of the projected 3D corners
against the annotated 2D bbox on the same label row: the median IoU over non-truncated
boxes is 0.97 (sequence 0000) and 0.98 (sequence 0011).

## Format notes

- Labels have 17 fields, the dimensions come in `h w l` order, and `(x, y, z)` is the
  center of the **bottom** face of the box in rectified camera coordinates (Y points down).
- `DontCare` rows carry track_id -1 and dummy 3D fields, so they are dropped.
- The tracking calib files use the keys `P0 P1 P2 P3 R_rect Tr_velo_cam Tr_imu_velo`,
  which differ from the object detection devkit. Only `P2` is needed.
- The boxes already live in rectified camera coordinates, so projecting is just
  `P2 @ [X, Y, Z, 1]`, with no `R_rect`.
