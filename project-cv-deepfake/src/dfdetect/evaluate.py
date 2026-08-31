"""Stage 3 — Evaluate on the held-out test set (never seen during training/model selection).

Loads the best checkpoint from train.py and scores it on test.zip's images — a
genuinely separate download from the train/val pool (val.zip), not just a random
split of the same file. That's the strongest leakage guard available without a
dedicated third data source.
"""

from __future__ import annotations

import json

import torch
from torch.utils.data import DataLoader

from dfdetect.config import load_config, resolve
from dfdetect.data import list_items, prepare
from dfdetect.dataset import FaceDataset
from dfdetect.metrics import binary_metrics
from dfdetect.modeling import build_model, pick_device
from dfdetect.train import _scores


def run(cfg=None) -> dict:
    cfg = cfg or load_config()
    mcfg, tcfg = cfg["model"], cfg["train"]
    device = pick_device(mcfg["device"])

    paths = prepare(cfg)
    test_items = list_items(paths["test"])
    test_loader = DataLoader(FaceDataset(test_items, mcfg["img_size"], augment=False),
                             batch_size=tcfg["batch_size"], shuffle=False)

    model = build_model(mcfg["backbone"]).to(device)
    ckpt = resolve(tcfg["ckpt_dir"]) / "best.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    print(f"[eval] loaded {ckpt} | test set: {len(test_items)} images "
          f"(held out — separate download from train/val)")

    y_true, y_score = _scores(model, test_loader, device)
    metrics = binary_metrics(y_true, y_score, cfg["eval"]["threshold"])
    print(f"[eval] test ROC-AUC={metrics['roc_auc']:.4f}  accuracy={metrics['accuracy']:.4f}  "
          f"F1={metrics['f1']:.4f}")
    print(f"[eval] confusion matrix: {metrics['confusion_matrix']}")

    out = resolve("data/reports") / "test_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print(f"[eval] wrote {out}")
    return metrics


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
