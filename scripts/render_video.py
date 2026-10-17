"""Render a KITTI tracking sequence to mp4 with the ground truth 3D boxes."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kitti_mot.dataset import group_by_frame, load_labels, load_p2
from kitti_mot.viz import write_video

TRAINING = ROOT / "data" / "kitti_tracking" / "training"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default="0000")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    frames = sorted((TRAINING / "image_02" / args.sequence).glob("*.png"))
    if not frames:
        raise SystemExit(f"no images for sequence {args.sequence}: run scripts/download_subset.py")
    gt = group_by_frame(load_labels(TRAINING / "label_02" / f"{args.sequence}.txt"))
    p2 = load_p2(TRAINING / "calib" / f"{args.sequence}.txt")

    out_path = Path(args.out) if args.out else ROOT / "out" / f"kitti_{args.sequence}_gt.mp4"
    boxes_by_frame = {frame: [(b.track_id, b) for b in boxes] for frame, boxes in gt.items()}
    drawn, skipped = write_video(frames, boxes_by_frame, p2, out_path, args.fps)

    print(f"{len(frames)} frames, {drawn} boxes drawn ({skipped} behind the camera)")
    print(out_path)


if __name__ == "__main__":
    main()
