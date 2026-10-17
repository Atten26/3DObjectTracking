# 3D Multi-Object Tracking on KITTI

Step 1: download a subset of the KITTI tracking benchmark and project the ground
truth 3D boxes onto the images.

Step 2: a constant-velocity Kalman tracker that rebuilds object identities from
geometry alone, plus the metrics to tell whether it worked.

Step 3: make the fake detector fail the way real ones do - stubbornly - and find
out what that costs.

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

# failures that persist, instead of flickering on and off every frame
venv/bin/python scripts/run_tracking.py --sequence 0000 --miss-rate 0.2 \
    --false-positives 1.0 --occlusion-frames 5 --ghost-frames 5
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

## Persistent failures

Independent per-frame noise is the easy case and it flatters the tracker. A real
detector loses an object for as long as the truck in front stays in front, and
hallucinates the same bush in every frame. `--occlusion-frames` and `--ghost-frames`
set how long one failure lasts; at 1 they reproduce the independent case.

The rates stay totals, so lengthening a failure makes it fire proportionally less
often. Otherwise the comparison would be between less noise and more of it rather
than between two shapes of the same noise. On short tracks a long occlusion gets cut
off when the object leaves the scene, so the persistent runs actually carry slightly
*less* missing data (14% against 20% at length 20) - which makes the damage below a
lower bound.

**`min_hits` stops protecting you.** Against independent false positives it is worth
99%; against ones that persist it is worth almost nothing, because a ghost that holds
still for three frames collects three hits and gets confirmed exactly like a real
object would. Same 1.0 false positives per frame throughout:

| ghost lifetime | false positives reported |
|---|---|
| 1 frame (independent) | 2.1 |
| 3 frames | 80.0 |
| 10 frames | 112.8 |
| 20 frames | 131.0 |

**The dangerous occlusion is the one just longer than `max_age`,** not the longest one.
Identity switches peak when occlusion length is near `max_age` and fall away on both
sides: below it the track coasts through, above it the events are rare enough not to
matter. The peak moves when `max_age` moves, which is what tells us it is the cause
and not a coincidence:

| occlusion | `max_age=1` | `max_age=3` | `max_age=10` | `max_age=20` |
|---|---|---|---|---|
| ~1 frame | **0.505** | 0.831 | 0.859 | 0.859 |
| ~3 frames | 0.523 | **0.638** | 0.854 | 0.858 |
| ~5 frames | 0.576 | 0.646 | 0.775 | 0.814 |
| ~10 frames | 0.659 | 0.704 | **0.759** | 0.803 |
| ~20 frames | 0.780 | 0.785 | 0.802 | **0.815** |

(IDF1, sequence 0000, 8 seeds. Bold marks each column's worst case.)

**`max_age` is not free either, and its bill arrives as identity switches.** On
perfectly clean input, raising it from 1 to 10 takes IDF1 on sequence 0000 from 1.000
down to 0.949. False positives stay at zero - a coasting track is not reported until
it is seen again - so the damage is invisible in that column and shows up entirely in
the identities. When an object legitimately leaves the scene its track keeps coasting,
and at ~0.6 m/frame it covers 6 m in ten frames, more than the 5.7 m median gap between
objects, so it drifts onto a neighbour and steals its detection. That coasting velocity
is mostly ego-motion, which makes this the same problem as the one below wearing a
different hat.

| | clean input | occlusions ~5 frames |
|---|---|---|
| `max_age=1` | **IDF1 1.000** | IDF1 0.576 |
| `max_age=10` | IDF1 0.949 | **IDF1 0.775** |

The default is 10, because a detector that never loses anything is not a thing that
exists, and 0.05 of IDF1 on an input we will never see is worth 0.20 on one we will.
On sequence 0011 it takes identity switches under persistent occlusion from 83.6 to
39.4. The honest fix is not to tune this number but to stop a coasting track from
competing for detections on equal terms with a track that can actually see its object.
