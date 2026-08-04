"""mAP@50 correctness tests — pure numpy, no model/dataset, so CI stays fast.

These pin down the metric that scores the untiled-vs-SAHI head-to-head; the model,
training, eval, and serving paths are verified locally (they need the 1.6GB VisDrone
tree + torch/ultralytics).
"""

from __future__ import annotations

import numpy as np

from overwatch.metrics import compute_map50, iou_matrix


def test_iou_identical_and_disjoint():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    assert iou_matrix(a, a)[0, 0] == 1.0
    b = np.array([[20, 20, 30, 30]], dtype=np.float32)
    assert iou_matrix(a, b)[0, 0] == 0.0


def test_iou_half_overlap():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)   # area 100
    b = np.array([[5, 0, 15, 10]], dtype=np.float32)   # area 100, overlap 50
    # inter=50, union=150 -> 1/3
    assert abs(iou_matrix(a, b)[0, 0] - (50 / 150)) < 1e-6


def test_perfect_predictions_map1():
    gt = np.array([[0, 0, 10, 10, 0], [20, 20, 30, 30, 1]], dtype=np.float32)
    preds = np.array([[0, 0, 10, 10, 0.9, 0], [20, 20, 30, 30, 0.8, 1]], dtype=np.float32)
    out = compute_map50([preds], [gt], num_classes=2)
    assert out["map50"] == 1.0
    assert out["n_classes_scored"] == 2


def test_no_predictions_map0():
    gt = np.array([[0, 0, 10, 10, 0]], dtype=np.float32)
    out = compute_map50([np.zeros((0, 6), dtype=np.float32)], [gt], num_classes=1)
    assert out["map50"] == 0.0


def test_wrong_class_scores_zero():
    gt = np.array([[0, 0, 10, 10, 0]], dtype=np.float32)
    preds = np.array([[0, 0, 10, 10, 0.9, 1]], dtype=np.float32)  # right box, wrong class
    out = compute_map50([preds], [gt], num_classes=2)
    assert out["map50"] == 0.0  # class 0 has GT but no correct det; class 1 has no GT


def test_low_iou_is_false_positive():
    gt = np.array([[0, 0, 10, 10, 0]], dtype=np.float32)
    preds = np.array([[7, 7, 17, 17, 0.9, 0]], dtype=np.float32)  # IoU well below 0.5
    out = compute_map50([preds], [gt], num_classes=1)
    assert out["map50"] == 0.0
