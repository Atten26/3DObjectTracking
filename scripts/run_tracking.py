"""Run the tracker on a KITTI sequence and report how well it recovered the identities.

The tracker is fed detections with their track_id stripped, so it has to rebuild
the identities from geometry alone. Use the noise options to find out what happens
once the input stops being perfect.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kitti_mot.dataset import group_by_frame, load_labels, load_p2
from kitti_mot.detections import Detector
from kitti_mot.metrics import evaluate
from kitti_mot.tracker import SIGMA_MEASURE, Tracker
from kitti_mot.viz import write_video

TRAINING = ROOT / "data" / "kitti_tracking" / "training"


def run(sequence, miss_rate=0.0, false_positives=0.0, position_noise=0.0, seed=0,
        min_hits=3, max_age=10, measurement_noise=None,
        occlusion_frames=1.0, ghost_frames=1.0):
    """Returns (metrics, predictions by frame).

    measurement_noise defaults to whatever noise we are injecting, which is the
    honest choice: the filter should be told how good its input really is.
    """
    gt = group_by_frame(load_labels(TRAINING / "label_02" / f"{sequence}.txt"))
    detector = Detector(np.random.default_rng(seed), miss_rate, false_positives,
                        position_noise, occlusion_frames, ghost_frames)
    if measurement_noise is None:
        measurement_noise = max(position_noise, SIGMA_MEASURE)
    tracker = Tracker(min_hits=min_hits, max_age=max_age, measurement_noise=measurement_noise)

    predictions = {}
    for frame in sorted(gt):
        predictions[frame] = tracker.update(detector(gt[frame]))

    truth = {frame: [(b.track_id, b) for b in boxes] for frame, boxes in gt.items()}
    return evaluate(truth, predictions), predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default="0000")
    parser.add_argument("--miss-rate", type=float, default=0.0)
    parser.add_argument("--false-positives", type=float, default=0.0)
    parser.add_argument("--position-noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument("--max-age", type=int, default=10)
    parser.add_argument("--measurement-noise", type=float, default=None)
    parser.add_argument("--occlusion-frames", type=float, default=1.0)
    parser.add_argument("--ghost-frames", type=float, default=1.0)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()

    results, predictions = run(
        args.sequence, args.miss_rate, args.false_positives, args.position_noise,
        args.seed, args.min_hits, args.max_age, args.measurement_noise,
        args.occlusion_frames, args.ghost_frames,
    )
    for name, value in results.items():
        print(f"  {name:16s} {value:.3f}" if isinstance(value, float) else f"  {name:16s} {value}")

    if args.video:
        frames = sorted((TRAINING / "image_02" / args.sequence).glob("*.png"))
        p2 = load_p2(TRAINING / "calib" / f"{args.sequence}.txt")
        out_path = ROOT / "out" / f"kitti_{args.sequence}_tracked.mp4"
        drawn, skipped = write_video(frames, predictions, p2, out_path)
        print(f"\n{len(frames)} frames, {drawn} boxes drawn ({skipped} behind the camera)")
        print(out_path)


if __name__ == "__main__":
    main()
