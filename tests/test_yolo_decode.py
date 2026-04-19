"""Unit tests for tools/npu_ref/yolo_decode.py (F1-T2)."""

from __future__ import annotations

import numpy as np
import pytest

from tools.npu_ref.yolo_decode import (
    Detection,
    box_iou,
    decode_yolov8_output,
    detection_match_rate,
    match_detections,
)


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------
def test_box_iou_identical():
    b = (10.0, 20.0, 30.0, 40.0)
    assert box_iou(b, b) == 1.0


def test_box_iou_disjoint():
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_box_iou_half_overlap():
    # Two unit squares offset by 0.5 along x — overlap = 0.5, union = 1.5
    iou = box_iou((0, 0, 1, 1), (0.5, 0, 1.5, 1))
    assert abs(iou - (0.5 / 1.5)) < 1e-6


def test_box_iou_zero_area_safe():
    # Degenerate box with zero area: IoU well-defined as 0.
    assert box_iou((0, 0, 0, 0), (1, 1, 2, 2)) == 0.0


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------
def _make_output(bboxes_xywh, class_scores):
    """Build a synthetic (1, 84, N) tensor from Python lists for testing."""
    N = len(bboxes_xywh)
    raw = np.zeros((1, 84, N), dtype=np.float32)
    for i, (cx, cy, w, h) in enumerate(bboxes_xywh):
        raw[0, 0, i] = cx
        raw[0, 1, i] = cy
        raw[0, 2, i] = w
        raw[0, 3, i] = h
        for cid, score in class_scores[i]:
            raw[0, 4 + cid, i] = score
    return raw


def test_decode_single_high_score_detection():
    raw = _make_output(
        bboxes_xywh=[(100.0, 100.0, 50.0, 50.0)],
        class_scores=[[(3, 0.9)]],
    )
    dets = decode_yolov8_output(raw)
    assert len(dets) == 1
    d = dets[0]
    assert d.class_id == 3
    assert d.score == pytest.approx(0.9, abs=1e-5)
    assert d.bbox_xyxy == (75.0, 75.0, 125.0, 125.0)


def test_decode_drops_below_threshold():
    raw = _make_output(
        bboxes_xywh=[(100.0, 100.0, 50.0, 50.0)],
        class_scores=[[(3, 0.1)]],     # below default 0.25
    )
    assert decode_yolov8_output(raw) == []


def test_decode_nms_suppresses_overlapping_same_class():
    # Three boxes, same class, heavily overlapping — NMS should keep
    # only the highest-scoring one.
    raw = _make_output(
        bboxes_xywh=[
            (100, 100, 50, 50),     # top score
            (101, 101, 50, 50),     # heavy IoU with above, lower score
            (200, 200, 50, 50),     # disjoint
        ],
        class_scores=[[(5, 0.95)], [(5, 0.9)], [(5, 0.8)]],
    )
    dets = decode_yolov8_output(raw)
    classes_seen = sorted(d.class_id for d in dets)
    assert classes_seen == [5, 5]          # two kept (top + disjoint)
    # The middle, heavily-overlapping one should have been suppressed.
    scores = sorted([d.score for d in dets], reverse=True)
    assert scores[0] == pytest.approx(0.95, abs=1e-5)
    assert scores[1] == pytest.approx(0.8, abs=1e-5)


def test_decode_nms_preserves_across_classes():
    # Heavily overlapping boxes but different classes — NMS should
    # keep both (per-class).
    raw = _make_output(
        bboxes_xywh=[(100, 100, 50, 50), (100, 100, 50, 50)],
        class_scores=[[(3, 0.9)], [(7, 0.9)]],
    )
    dets = decode_yolov8_output(raw)
    assert len(dets) == 2
    assert sorted(d.class_id for d in dets) == [3, 7]


def test_decode_accepts_bare_2d_input():
    raw = _make_output(
        bboxes_xywh=[(50, 50, 20, 20)],
        class_scores=[[(0, 0.5)]],
    )
    # Strip batch dim.
    dets = decode_yolov8_output(raw[0])
    assert len(dets) == 1


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
def test_match_all_detections_agree():
    ref = [Detection(0, 0.9, (0, 0, 20, 20)),
           Detection(1, 0.8, (40, 40, 60, 60))]
    # Predicted slightly perturbed (IoU ≈ 0.81, above 0.7 threshold)
    # and same class.
    pred = [Detection(0, 0.9, (1, 1, 21, 21)),
            Detection(1, 0.8, (40, 40, 60, 60))]
    matched, missed, total = match_detections(
        ref, pred, iou_threshold=0.7)
    assert matched == 2 and missed == 0 and total == 2


def test_match_rejects_wrong_class():
    ref = [Detection(0, 0.9, (0, 0, 10, 10))]
    pred = [Detection(1, 0.9, (0, 0, 10, 10))]   # right box, wrong class
    matched, missed, _ = match_detections(
        ref, pred, iou_threshold=0.7)
    assert matched == 0 and missed == 1


def test_match_rejects_low_iou():
    ref = [Detection(0, 0.9, (0, 0, 10, 10))]
    pred = [Detection(0, 0.9, (100, 100, 110, 110))]
    matched, _, _ = match_detections(ref, pred, iou_threshold=0.7)
    assert matched == 0


def test_match_each_pred_at_most_once():
    """One prediction can't satisfy two references."""
    ref = [Detection(0, 0.9, (0, 0, 10, 10)),
           Detection(0, 0.9, (0, 0, 10, 10))]
    pred = [Detection(0, 0.9, (0, 0, 10, 10))]
    matched, missed, total = match_detections(
        ref, pred, iou_threshold=0.7)
    assert matched == 1 and missed == 1 and total == 2


def test_match_rate_empty_reference_is_perfect():
    """No reference detections → match rate = 1.0 (nothing to miss)."""
    assert detection_match_rate([], [Detection(0, 0.9, (0, 0, 1, 1))]) == 1.0
