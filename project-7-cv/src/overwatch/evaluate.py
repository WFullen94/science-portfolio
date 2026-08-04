"""Stage 3 — the head-to-head: plain (untiled) vs SAHI-tiled inference.

VisDrone is dominated by tiny objects. A detector run once on the whole (downscaled)
frame misses many of them; **SAHI** slices the frame into overlapping tiles, runs the
detector on each, and merges — recovering small objects at the cost of more forward
passes (latency). This module quantifies that trade the honest way: the *same* model
and the *same* mAP@50 implementation ([metrics.py](metrics.py)) score both paths, plus
small-object recall and per-image latency so the cost is on the table too.
"""

from __future__ import annotations

import json
import time

import numpy as np
from PIL import Image

from overwatch.config import load_config, resolve
from overwatch.metrics import compute_map50, iou_matrix
from overwatch.utils import pick_device


def _weights(cfg):
    best = resolve(cfg["serve"]["weights"])
    return str(best) if best.exists() else cfg["model"]["weights"]


def _val_items(cfg, limit):
    import yaml
    ds_yaml = resolve(cfg["dataset"]["yaml"])
    spec = yaml.safe_load(ds_yaml.read_text())
    root = (ds_yaml.parent / spec.get("path", ".")).resolve()
    img_dir = root / spec.get("val", "images/val")
    lbl_dir = root / spec.get("val", "images/val").replace("images", "labels")
    imgs = sorted(p for p in img_dir.glob("*")
                  if p.suffix.lower() in {".jpg", ".png", ".jpeg"})[:limit]
    return [(p, lbl_dir / f"{p.stem}.txt") for p in imgs]


def _load_gt(lbl_path, w, h) -> np.ndarray:
    if not lbl_path.exists():
        return np.zeros((0, 5), dtype=np.float32)
    rows = []
    for line in lbl_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        c, cx, cy, bw, bh = (float(x) for x in parts[:5])
        rows.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                     (cx + bw / 2) * w, (cy + bh / 2) * h, c])
    return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 5), dtype=np.float32)


def _small_recall(preds_per_img, gts_per_img, small_area: float, iou_thr=0.5) -> float:
    """Recall on GT boxes smaller than `small_area` px² — SAHI's target regime."""
    matched, total = 0, 0
    for preds, gts in zip(preds_per_img, gts_per_img):
        if len(gts) == 0:
            continue
        areas = (gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1])
        small = gts[areas < small_area]
        total += len(small)
        if len(small) == 0 or len(preds) == 0:
            continue
        ious = iou_matrix(small[:, :4], preds[:, :4])
        same_cls = small[:, 4][:, None] == preds[:, 5][None, :]
        hit = ((ious >= iou_thr) & same_cls).any(axis=1)
        matched += int(hit.sum())
    return matched / total if total else 0.0


def _predict_untiled(model, path, cfg, device):
    r = model.predict(str(path), conf=cfg["eval"]["conf"], imgsz=cfg["model"]["imgsz"],
                      device=device, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    b = r.boxes
    return np.concatenate([b.xyxy.cpu().numpy(),
                           b.conf.cpu().numpy()[:, None],
                           b.cls.cpu().numpy()[:, None]], axis=1).astype(np.float32)


def _predict_sahi(det_model, path, scfg):
    from sahi.predict import get_sliced_prediction
    res = get_sliced_prediction(
        str(path), det_model,
        slice_height=scfg["slice_height"], slice_width=scfg["slice_width"],
        overlap_height_ratio=scfg["overlap_ratio"], overlap_width_ratio=scfg["overlap_ratio"],
        verbose=0,
    )
    rows = []
    for op in res.object_prediction_list:
        x1, y1, x2, y2 = op.bbox.to_xyxy()
        rows.append([x1, y1, x2, y2, op.score.value, op.category.id])
    return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 6), dtype=np.float32)


def run(cfg=None) -> dict:
    from sahi import AutoDetectionModel
    from ultralytics import YOLO

    cfg = cfg or load_config()
    device = pick_device(cfg["model"]["device"])
    weights = _weights(cfg)
    items = _val_items(cfg, cfg["eval"]["max_images"])
    nc = cfg["dataset"]["nc"]
    print(f"[eval] weights={weights} device={device} images={len(items)}")

    model = YOLO(weights)
    sahi_device = "cuda:0" if device == "0" else device
    det_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=weights,
        confidence_threshold=cfg["eval"]["conf"], device=sahi_device,
    )

    # Warm up both paths on the first image (untimed) so MPS/graph-compile cost
    # doesn't land on the untiled path's first measurement — makes latency fair.
    _predict_untiled(model, items[0][0], cfg, device)
    _predict_sahi(det_model, items[0][0], cfg["eval"]["sahi"])

    gts, un_preds, sa_preds = [], [], []
    un_time = sa_time = 0.0
    for img_path, lbl_path in items:
        w, h = Image.open(img_path).size
        gts.append(_load_gt(lbl_path, w, h))
        t = time.perf_counter(); un_preds.append(_predict_untiled(model, img_path, cfg, device)); un_time += time.perf_counter() - t
        t = time.perf_counter(); sa_preds.append(_predict_sahi(det_model, img_path, cfg["eval"]["sahi"])); sa_time += time.perf_counter() - t

    small_area = float(cfg["eval"].get("small_area_px", 1024))  # 32x32 px
    un = {"map50": compute_map50(un_preds, gts, nc)["map50"],
          "small_recall": _small_recall(un_preds, gts, small_area),
          "ms_per_img": 1000 * un_time / len(items)}
    sa = {"map50": compute_map50(sa_preds, gts, nc)["map50"],
          "small_recall": _small_recall(sa_preds, gts, small_area),
          "ms_per_img": 1000 * sa_time / len(items)}

    print("\n[eval] head-to-head — untiled vs SAHI-tiled (val, held-out):\n")
    print("| method         | mAP@50 | small-obj recall | ms/img |")
    print("|---|---|---|---|")
    print(f"| Untiled        | {un['map50']:.4f} | {un['small_recall']:.4f} | {un['ms_per_img']:.0f} |")
    print(f"| SAHI-tiled     | {sa['map50']:.4f} | {sa['small_recall']:.4f} | {sa['ms_per_img']:.0f} |")
    print(f"\n[eval] SAHI Δ mAP@50 = {sa['map50'] - un['map50']:+.4f} | "
          f"Δ small-recall = {sa['small_recall'] - un['small_recall']:+.4f} | "
          f"latency ×{sa['ms_per_img'] / max(un['ms_per_img'], 1e-6):.1f}")

    out = resolve("data/reports") / "sahi_headtohead.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n_images": len(items), "untiled": un, "sahi": sa}, indent=2))
    print(f"[eval] wrote {out}")
    return {"untiled": un, "sahi": sa}


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
