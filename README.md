# 3D Multi-Object Tracking on KITTI

Step 1: download a subset of the KITTI tracking benchmark and project the ground
truth 3D boxes onto the images.

Step 2: a constant-velocity Kalman tracker that rebuilds object identities from
geometry alone, plus the metrics to tell whether it worked.

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
venv/bin/python scripts/render_video.py --sequence 0000   # ground truth -> out/kitti_0000_gt.mp4
venv/bin/python scripts/smoke_check.py

# tracking: detections are the GT boxes with their track_id stripped
venv/bin/python scripts/run_tracking.py --sequence 0000
venv/bin/python scripts/run_tracking.py --sequence 0000 --miss-rate 0.1 \
    --false-positives 0.5 --position-noise 0.25 --video
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

## Tracking

The tracker is deliberately plain: a constant-velocity Kalman filter on the 3D center,
Hungarian assignment on euclidean distance, and a track lifecycle with `min_hits` and
`max_age`. Parameters come from the labels themselves - objects never move more than
2.47 m between frames, and the 95th percentile of their acceleration is 0.12 m/frame^2.

On perfect detections it is exact: MOTA 1.000, IDF1 1.000, zero identity switches on
sequence 0000. That is the uninteresting case. What the numbers below actually say:

**`min_hits` is a bet, not a free win.** Holding a track back for 2 frames costs exactly
one missed object per track and nothing else, as long as the detector is clean. Its value
shows up only against false positives, and there it is decisive:

| false positives/frame | `min_hits=1` | `min_hits=3` |
|---|---|---|
| 0.5 | 71 leak through | 0.4 |
| 2.0 | 301 leak through | 9.2 |

**The filter has to be told how bad its input is.** `SIGMA_MEASURE` is the standard
deviation the filter assumes for each detection. Leaving it at the ground truth value
of 0.10 m while feeding it 0.5 m of real noise costs 0.18 IDF1 and triples the identity
switches. `scripts/run_tracking.py` therefore defaults it to the noise it is injecting.

**Past 1 m of position noise, the metric stops being able to see.** At that level 26.7%
of detections land further from their object than the 2 m matching threshold, so even an
oracle tracker that already knows every identity scores MOTA 0.466. The tracker scores
0.487 there, i.e. slightly better than the oracle, because the Kalman filter smooths
noise that the oracle takes at face value. Low scores in that regime say more about the
measurement than about the tracking.

## Known limitation: ego-motion

The state lives in camera coordinates, and the camera is on a moving car, so a parked
object still carries a velocity here. In sequence 0000 most objects share a mean
displacement of about `(0, 0, -0.6)` m/frame, which is not them moving, it is the ego
vehicle driving forward at ~6 m/s. One of them is a *pedestrian*, who would otherwise
be running at 8 m/s.

Constant velocity absorbs this while the ego vehicle goes straight at a steady speed,
and breaks when it turns or brakes. Compensating properly means pulling in the IMU/GPS
`oxts` data and expressing the state in world coordinates. Not done yet, on purpose:
the cost of not doing it is now measurable.
