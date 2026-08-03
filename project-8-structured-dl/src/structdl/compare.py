"""Stage 1c — the head-to-head.

Runs each model in its OWN subprocess and collects the results. Two reasons:

  1. Fair comparison — `build_bundle` is deterministic (fixed seed + cached
     parquet), so every process reconstructs the *identical* split. The models
     never need to share an in-memory bundle to be comparing the same rows.
  2. Stability — XGBoost and PyTorch each ship their own OpenMP runtime; co-resident
     in one process, whichever runs its threaded kernels second segfaults on macOS.
     Separate processes never co-reside, so both run clean. (See structdl/__init__.)

The honest question this answers isn't "does deep learning win?" — the tabular
literature (Grinsztajn et al., 2022) says trees usually still do. It's *how close
does a Transformer get, and where*. State the result plainly, whichever way it lands.
"""

from __future__ import annotations

import json
import subprocess
import sys

from structdl.config import PROJECT_ROOT, load_config, resolve
from structdl.data import build_bundle

METRIC_ORDER = ["roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1"]


def _run_model(module: str) -> None:
    print(f"\n[compare] === {module} (isolated process) ===", flush=True)
    r = subprocess.run([sys.executable, "-u", "-m", module], cwd=PROJECT_ROOT)
    if r.returncode != 0:
        raise SystemExit(f"[compare] {module} failed (exit {r.returncode})")


def _row(name: str, m: dict) -> str:
    cells = " | ".join(f"{m[k]:.4f}" for k in METRIC_ORDER)
    return f"| {name:<16} | {cells} |"


def main() -> int:
    cfg = load_config()

    # Build (and cache) the shared split once so both subprocesses read the same
    # deterministic parquet and reconstruct the identical split.
    bundle = build_bundle(cfg)
    print(f"[compare] shared split: train={len(bundle.y_train)} "
          f"val={len(bundle.y_val)} test={len(bundle.y_test)} "
          f"| {len(bundle.num_cols)} numeric + {len(bundle.cat_cols)} categorical features")

    _run_model("structdl.xgb_baseline")
    _run_model("structdl.ft_transformer")

    reports = resolve(cfg["paths"]["reports"])
    xgb_metrics = json.loads((reports / "xgb_metrics.json").read_text())
    ft_metrics = json.loads((reports / "ft_metrics.json").read_text())

    header = "| model            | " + " | ".join(METRIC_ORDER) + " |"
    sep = "|" + "---|" * (len(METRIC_ORDER) + 1)
    print("\n[compare] head-to-head (held-out test set):\n")
    print(header)
    print(sep)
    print(_row("XGBoost", xgb_metrics))
    print(_row("FT-Transformer", ft_metrics))

    winner = ("FT-Transformer" if ft_metrics["roc_auc"] > xgb_metrics["roc_auc"]
              else "XGBoost")
    delta = abs(ft_metrics["roc_auc"] - xgb_metrics["roc_auc"])
    print(f"\n[compare] higher test ROC-AUC: {winner} (Δ={delta:.4f})")

    out = reports / "headtohead.json"
    out.write_text(json.dumps({
        "dataset": cfg["dataset"]["name"],
        "n_train": len(bundle.y_train), "n_test": len(bundle.y_test),
        "n_numeric": len(bundle.num_cols), "n_categorical": len(bundle.cat_cols),
        "xgboost": xgb_metrics, "ft_transformer": ft_metrics,
        "higher_roc_auc": winner, "roc_auc_delta": delta,
    }, indent=2))
    print(f"[compare] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
