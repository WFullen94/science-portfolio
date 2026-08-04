"""Stage 2c — the sequence head-to-head: time-series Transformer vs GRU.

One shared window set (built once, identical for both), both trained with the same
loop, scored on the same held-out test set. Unlike stage 1, both models are pure
torch — no XGBoost, so no OpenMP clash — and they run in one process on one bundle.

Same honest framing as stage 1: report which architecture wins on this temporal-burst
detection task, and by how much. Whichever way it lands.
"""

from __future__ import annotations

import json

from structdl import gru_baseline, ts_transformer
from structdl.config import load_config, resolve
from structdl.sequence_data import build_sequence_bundle

METRIC_ORDER = ["roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1"]


def _row(name: str, m: dict) -> str:
    return f"| {name:<16} | " + " | ".join(f"{m[k]:.4f}" for k in METRIC_ORDER) + " |"


def main() -> int:
    cfg = load_config()
    b = build_sequence_bundle(cfg)
    print(f"[ts-compare] shared windows: train={len(b.ytrain)} val={len(b.yval)} "
          f"test={len(b.ytest)} | (T={b.seq_len}, F={len(b.features)})")

    print("\n[ts-compare] === GRU baseline ===")
    gru_metrics = gru_baseline.run(b, cfg)
    print("\n[ts-compare] === time-series Transformer ===")
    ts_metrics = ts_transformer.run(b, cfg)

    header = "| model            | " + " | ".join(METRIC_ORDER) + " |"
    sep = "|" + "---|" * (len(METRIC_ORDER) + 1)
    print("\n[ts-compare] head-to-head (held-out test set):\n")
    print(header)
    print(sep)
    print(_row("GRU", gru_metrics))
    print(_row("TS-Transformer", ts_metrics))

    winner = ("TS-Transformer" if ts_metrics["roc_auc"] > gru_metrics["roc_auc"]
              else "GRU")
    delta = abs(ts_metrics["roc_auc"] - gru_metrics["roc_auc"])
    print(f"\n[ts-compare] higher test ROC-AUC: {winner} (Δ={delta:.4f})")

    out = resolve(cfg["paths"]["reports"]) / "ts_headtohead.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "task": "sequence intrusion detection (temporal burst)",
        "seq_len": b.seq_len, "n_features": len(b.features),
        "n_train": len(b.ytrain), "n_test": len(b.ytest),
        "gru": gru_metrics, "ts_transformer": ts_metrics,
        "higher_roc_auc": winner, "roc_auc_delta": delta,
    }, indent=2))
    print(f"[ts-compare] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
