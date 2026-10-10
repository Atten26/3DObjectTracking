"""Render a KITTI tracking sequence to mp4 with the ground truth 3D boxes."""

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kitti_mot.dataset import group_by_frame, load_labels, load_p2
from kitti_mot.viz import draw_frame

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
    by_frame = group_by_frame(load_labels(TRAINING / "label_02" / f"{args.sequence}.txt"))
    p2 = load_p2(TRAINING / "calib" / f"{args.sequence}.txt")

    out_path = Path(args.out) if args.out else ROOT / "out" / f"kitti_{args.sequence}_gt.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # the codec wants even dimensions, otherwise it silently drops a row
    height, width = (n - n % 2 for n in cv2.imread(str(frames[0])).shape[:2])
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height))

    drawn = skipped = 0
    for path in frames:
        image = cv2.imread(str(path))[:height, :width]
        if image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height))
        boxes = by_frame.get(int(path.stem), [])
        skipped += draw_frame(image, boxes, p2)
        drawn += len(boxes)
        writer.write(image)
    writer.release()

    print(f"{len(frames)} frames, {drawn - skipped} boxes drawn ({skipped} behind the camera)")
    print(out_path)


if __name__ == "__main__":
    main()
