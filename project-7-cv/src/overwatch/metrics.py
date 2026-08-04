"""A compact, self-contained mAP@50 — so untiled and SAHI-tiled detections are scored
by the *same* implementation (a fair head-to-head, not two different mAP definitions).

VOC-style: greedy IoU matching at threshold 0.5, all-points AP per class, mean over
classes that have ground truth. Inputs are pixel-space xyxy boxes.
"""

from __future__ import annotations

import numpy as np


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """IoU between every box in A (M,4) and B (N,4) -> (M,N)."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    tl = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    br = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = np.clip(br - tl, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def _ap_all_points(recall: np.ndarray, precision: np.ndarray) -> float:
    """All-points (VOC2010+) average precision: area under the P-R envelope."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])  # precision envelope (monotone)
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def compute_map50(preds_per_img, gts_per_img, num_classes: int, iou_thr: float = 0.5) -> dict:
    """preds_per_img[i]: (K,6) [x1,y1,x2,y2,score,cls]; gts_per_img[i]: (G,5) [x1,y1,x2,y2,cls]."""
    aps = {}
    for c in range(num_classes):
        scores, tps, fps, n_gt = [], [], [], 0
        for preds, gts in zip(preds_per_img, gts_per_img):
            gt_c = gts[gts[:, 4] == c][:, :4] if len(gts) else np.zeros((0, 4))
            n_gt += len(gt_c)
            det_c = preds[preds[:, 5] == c] if len(preds) else np.zeros((0, 6))
            if len(det_c) == 0:
                continue
            order = np.argsort(-det_c[:, 4])
            det_c = det_c[order]
            matched = np.zeros(len(gt_c), dtype=bool)
            ious = iou_matrix(det_c[:, :4], gt_c)
            for di in range(len(det_c)):
                scores.append(det_c[di, 4])
                if len(gt_c) == 0:
                    tps.append(0); fps.append(1); continue
                gi = int(np.argmax(ious[di]))
                if ious[di, gi] >= iou_thr and not matched[gi]:
                    matched[gi] = True; tps.append(1); fps.append(0)
                else:
                    tps.append(0); fps.append(1)
        if n_gt == 0:
            continue  # class absent from GT — excluded from mAP
        if not scores:
            aps[c] = 0.0; continue
        order = np.argsort(-np.array(scores))
        tp = np.cumsum(np.array(tps)[order])
        fp = np.cumsum(np.array(fps)[order])
        recall = tp / n_gt
        precision = tp / np.maximum(tp + fp, 1e-9)
        aps[c] = _ap_all_points(recall, precision)

    map50 = float(np.mean(list(aps.values()))) if aps else 0.0
    return {"map50": map50, "per_class_ap": aps, "n_classes_scored": len(aps)}
