"""YOLOv8 detection output decoder + NMS + IoU matcher (F1-T2).

YOLOv8's exported `output0` tensor has shape `(1, 84, 8400)`:
  - Channels 0..3  : bbox (cx, cy, w, h) in pixel units relative to the
                     640×640 input. Already decoded by the model's
                     Detect head (no anchor decoding needed here).
  - Channels 4..83 : per-class scores, already sigmoided (ranges [0,1]).
  - 8400 anchors   : 80² + 40² + 20² = 6400+1600+400 across the three
                     feature strides (8, 16, 32).

Pipeline:
    raw output → decode_yolov8_output → filter by score → NMS → [Detection]

Two detection lists (e.g. FP32 reference vs NPU path) are compared via
`match_detections`, which computes per-reference best-IoU matches and
returns counts: (# matched at threshold, # unmatched, total reference).
This yields the plan-specified F1-C5 acceptance metric: "IoU ≥ 0.7 on
80%+ of detections".

Pure numpy — no torch / torchvision / opencv dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Detection:
    class_id: int
    score: float
    bbox_xyxy: Tuple[float, float, float, float]   # (x1, y1, x2, y2)

    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _xywh_to_xyxy(cx: float, cy: float, w: float, h: float
                    ) -> Tuple[float, float, float, float]:
    return (cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5)


def box_iou(a: Tuple[float, float, float, float],
            b: Tuple[float, float, float, float]) -> float:
    """IoU of two xyxy bounding boxes. Zero for non-overlapping."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def _nms_per_class(dets: List[Detection], iou_threshold: float
                    ) -> List[Detection]:
    """Greedy NMS. Assumes dets are sorted descending by score."""
    kept: List[Detection] = []
    for d in dets:
        suppressed = False
        for k in kept:
            if box_iou(d.bbox_xyxy, k.bbox_xyxy) > iou_threshold:
                suppressed = True
                break
        if not suppressed:
            kept.append(d)
    return kept


def decode_yolov8_output(raw: np.ndarray,
                          *,
                          score_threshold: float = 0.25,
                          iou_threshold: float = 0.45,
                          max_detections: int = 300,
                          ) -> List[Detection]:
    """Decode a YOLOv8 `output0` tensor to a list of Detections.

    Args:
        raw: shape (1, 84, 8400) or (84, 8400). Channels 0..3 are xywh
            bbox (pixels on the 640×640 letterboxed input); channels
            4..83 are per-class sigmoided scores.
        score_threshold: drop anchors whose max-class score is below
            this (standard default 0.25).
        iou_threshold: NMS IoU threshold (standard 0.45).
        max_detections: final cap on detection count; defensively
            truncates low-score residuals.

    Returns:
        List of Detection, sorted by score descending after per-class
        NMS + cross-class merge.
    """
    if raw.ndim == 3:
        if raw.shape[0] != 1:
            raise ValueError(
                f"expected batch=1, got {raw.shape[0]}"
            )
        raw = raw[0]
    if raw.shape[0] != 84:
        raise ValueError(
            f"expected 84 channels, got {raw.shape[0]}"
        )
    bboxes_xywh = raw[:4]              # (4, 8400)
    class_scores = raw[4:]             # (80, 8400)
    best_scores = class_scores.max(axis=0)              # (8400,)
    best_classes = class_scores.argmax(axis=0)          # (8400,)
    keep_mask = best_scores > score_threshold
    if not np.any(keep_mask):
        return []

    kept_xywh = bboxes_xywh[:, keep_mask]
    kept_scores = best_scores[keep_mask]
    kept_classes = best_classes[keep_mask]

    # Build preliminary detection list sorted by score.
    order = np.argsort(-kept_scores)
    all_dets: List[Detection] = []
    for idx in order[:max_detections * 4]:  # keep some headroom for NMS
        cx, cy, w, h = kept_xywh[:, idx].tolist()
        all_dets.append(Detection(
            class_id=int(kept_classes[idx]),
            score=float(kept_scores[idx]),
            bbox_xyxy=_xywh_to_xyxy(cx, cy, w, h),
        ))

    # Per-class NMS, then merge.
    by_class: dict[int, List[Detection]] = {}
    for d in all_dets:
        by_class.setdefault(d.class_id, []).append(d)
    final: List[Detection] = []
    for cls, dets in by_class.items():
        final.extend(_nms_per_class(dets, iou_threshold))
    final.sort(key=lambda d: -d.score)
    return final[:max_detections]


def match_detections(reference: List[Detection],
                      predicted: List[Detection],
                      *,
                      iou_threshold: float = 0.7,
                      require_class_match: bool = True,
                      ) -> Tuple[int, int, int]:
    """For each reference detection, find a best-matching prediction.

    A prediction "matches" when:
      - `iou ≥ iou_threshold`, and
      - (if `require_class_match`) the class ids agree.
    Each prediction can match at most one reference (greedy, highest
    IoU first).

    Returns:
        (n_matched, n_missed, n_total)
        where n_total = len(reference).
    """
    n_total = len(reference)
    if n_total == 0:
        return (0, 0, 0)

    used = [False] * len(predicted)
    n_matched = 0
    for ref in reference:
        best_iou = 0.0
        best_idx = -1
        for j, p in enumerate(predicted):
            if used[j]:
                continue
            if require_class_match and p.class_id != ref.class_id:
                continue
            iou = box_iou(ref.bbox_xyxy, p.bbox_xyxy)
            if iou > best_iou:
                best_iou = iou
                best_idx = j
        if best_idx >= 0 and best_iou >= iou_threshold:
            used[best_idx] = True
            n_matched += 1
    return (n_matched, n_total - n_matched, n_total)


def detection_match_rate(reference: List[Detection],
                          predicted: List[Detection],
                          *,
                          iou_threshold: float = 0.7,
                          ) -> float:
    matched, _, total = match_detections(
        reference, predicted, iou_threshold=iou_threshold)
    return matched / total if total > 0 else 1.0
