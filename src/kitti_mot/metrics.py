"""CLEAR MOT and IDF1, computed on 3D center distance.

Two different questions, which is why both are here:
  MOTA  counts per-frame mistakes - a missed object, a spurious one, an identity
        that flipped. It is dominated by detection quality.
  IDF1  asks instead how much of each object's life was spent under the right
        identity, so a tracker that swaps two cars halfway through is punished
        far more than MOTA punishes it.
"""

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

THRESHOLD = 2.0  # m, a prediction further than this from a GT object is not that object


def _match(gt, pred, previous, threshold):
    """Greedy on last frame's pairs, Hungarian on the rest. Returns gt_id -> pred_id."""
    gt_ids = [i for i, _ in gt]
    pred_ids = [i for i, _ in pred]
    distance = np.linalg.norm(
        np.array([b.center for _, b in gt])[:, None] - np.array([b.center for _, b in pred])[None],
        axis=-1,
    )

    matched, used_gt, used_pred = {}, set(), set()
    # keeping a pair that still holds is what makes an identity switch detectable
    for g, gt_id in enumerate(gt_ids):
        pred_id = previous.get(gt_id)
        if pred_id in pred_ids:
            p = pred_ids.index(pred_id)
            if distance[g, p] <= threshold:
                matched[gt_id] = pred_id
                used_gt.add(g)
                used_pred.add(p)

    free_gt = [g for g in range(len(gt_ids)) if g not in used_gt]
    free_pred = [p for p in range(len(pred_ids)) if p not in used_pred]
    if free_gt and free_pred:
        sub = distance[np.ix_(free_gt, free_pred)]
        for r, c in zip(*linear_sum_assignment(sub)):
            if sub[r, c] <= threshold:
                matched[gt_ids[free_gt[r]]] = pred_ids[free_pred[c]]

    error = sum(
        distance[gt_ids.index(g), pred_ids.index(p)] for g, p in matched.items()
    )
    return matched, error


def evaluate(gt_by_frame, pred_by_frame, threshold=THRESHOLD):
    """Both sets are {frame: [(track_id, Box3D), ...]}."""
    previous = {}
    false_positives = misses = switches = matches = 0
    total_gt = total_pred = 0
    error = 0.0
    cooccurrence = defaultdict(int)
    gt_life, pred_life = defaultdict(int), defaultdict(int)

    for frame in sorted(set(gt_by_frame) | set(pred_by_frame)):
        gt = gt_by_frame.get(frame, [])
        pred = pred_by_frame.get(frame, [])
        total_gt += len(gt)
        total_pred += len(pred)
        for i, _ in gt:
            gt_life[i] += 1
        for i, _ in pred:
            pred_life[i] += 1

        if gt and pred:
            # every close pair counts towards IDF1, which does its own global matching
            for gt_id, gt_box in gt:
                for pred_id, pred_box in pred:
                    if np.linalg.norm(gt_box.center - pred_box.center) <= threshold:
                        cooccurrence[(gt_id, pred_id)] += 1
            frame_matches, frame_error = _match(gt, pred, previous, threshold)
            error += frame_error
        else:
            frame_matches = {}

        for gt_id, pred_id in frame_matches.items():
            if gt_id in previous and previous[gt_id] != pred_id:
                switches += 1
        previous.update(frame_matches)

        matches += len(frame_matches)
        misses += len(gt) - len(frame_matches)
        false_positives += len(pred) - len(frame_matches)

    return {
        "MOTA": 1.0 - (misses + false_positives + switches) / total_gt if total_gt else 0.0,
        "MOTP": error / matches if matches else float("nan"),
        "IDF1": _idf1(cooccurrence, gt_life, pred_life),
        "ID switches": switches,
        "false positives": false_positives,
        "misses": misses,
        "matches": matches,
        "GT boxes": total_gt,
    }


def _idf1(cooccurrence, gt_life, pred_life):
    """One identity per object for the whole sequence, chosen to maximise agreement."""
    total_gt, total_pred = sum(gt_life.values()), sum(pred_life.values())
    if not cooccurrence:
        return 0.0

    gt_ids = sorted(gt_life)
    pred_ids = sorted(pred_life)
    gain = np.zeros((len(gt_ids), len(pred_ids)))
    for (gt_id, pred_id), frames in cooccurrence.items():
        gain[gt_ids.index(gt_id), pred_ids.index(pred_id)] = frames

    rows, cols = linear_sum_assignment(-gain)
    true_positives = gain[rows, cols].sum()
    return 2 * true_positives / (total_gt + total_pred) if total_gt + total_pred else 0.0
