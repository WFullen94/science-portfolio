"""Stage 5 — the architecture head-to-head: 4 fine-tuned CNNs + a CLIP linear probe,
all on the identical data split (deterministic given the shared seed), all scored on
the identical held-out test set. Same "fair comparison" discipline as every other
head-to-head in this portfolio.
"""

from __future__ import annotations

import json

from dfdetect import clip_probe, evaluate, train
from dfdetect.config import load_config, resolve
from dfdetect.modeling import BACKBONES

METRIC_ORDER = ["roc_auc", "accuracy", "precision", "recall", "f1"]


def _row(name: str, m: dict) -> str:
    return f"| {name:<26} | " + " | ".join(f"{m[k]:.4f}" for k in METRIC_ORDER) + " |"


def main() -> int:
    cfg = load_config()
    results = {}

    for bb in BACKBONES:
        print(f"\n[compare] === {bb} (fine-tune) ===")
        train.run(cfg, backbone=bb)
        results[bb] = evaluate.run(cfg, backbone=bb)

    print("\n[compare] === clip_vit_b32_linear_probe (frozen features, no fine-tune) ===")
    results["clip_vit_b32_linear_probe"] = clip_probe.run(cfg)

    header = "| model | " + " | ".join(METRIC_ORDER) + " |"
    sep = "|" + "---|" * (len(METRIC_ORDER) + 1)
    print("\n[compare] architecture head-to-head (held-out test set):\n")
    print(header)
    print(sep)
    for name, m in results.items():
        print(_row(name, m))

    out = resolve("data/reports") / "architecture_headtohead.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[compare] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
