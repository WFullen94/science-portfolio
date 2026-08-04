"""Stage 2 (headline) — the length sweep: where does attention beat recurrence?

Runs the GRU vs time-series-Transformer head-to-head at several sequence lengths and
reports test ROC-AUC as a function of window length. The hypothesis (and the lesson):
the Transformer's advantage widens as the window grows, because the attack burst is a
short, absolutely-sized event that a last-hidden-state GRU increasingly forgets, while
attention reads it directly regardless of position.

A single "who won" number is anecdote; this curve is the actual finding.
"""

from __future__ import annotations

import copy
import json

from structdl import gru_baseline, ts_transformer
from structdl.config import load_config, resolve
from structdl.sequence_data import build_sequence_bundle


def _cfg_for_len(base, L: int):
    cfg = copy.deepcopy(base)
    cfg["sequence"]["seq_len"] = L
    cap = base["sequence"]["sweep_max_epochs"]
    for blk in ("ts_transformer", "gru"):
        cfg[blk]["max_epochs"] = cap
    return cfg


def main() -> int:
    base = load_config()
    lengths = base["sequence"]["sweep_seq_lens"]
    rows = []
    for L in lengths:
        cfg = _cfg_for_len(base, L)
        b = build_sequence_bundle(cfg)
        print(f"\n[sweep] === seq_len={L}  (train={len(b.ytrain)}, test={len(b.ytest)}) ===")
        gru = gru_baseline.run(b, cfg, log_to_mlflow=False)
        tst = ts_transformer.run(b, cfg, log_to_mlflow=False)
        rows.append({
            "seq_len": L,
            "gru_roc_auc": gru["roc_auc"], "ts_roc_auc": tst["roc_auc"],
            "gru_f1": gru["f1"], "ts_f1": tst["f1"],
            "ts_minus_gru_roc_auc": tst["roc_auc"] - gru["roc_auc"],
        })

    print("\n[sweep] ROC-AUC vs sequence length (held-out test):\n")
    print("| seq_len | GRU ROC-AUC | TS-Transformer ROC-AUC | Δ (TS − GRU) |")
    print("|---|---|---|---|")
    for r in rows:
        print(f"| {r['seq_len']:>3} | {r['gru_roc_auc']:.4f} | "
              f"{r['ts_roc_auc']:.4f} | {r['ts_minus_gru_roc_auc']:+.4f} |")

    trend = "widens" if rows[-1]["ts_minus_gru_roc_auc"] > rows[0]["ts_minus_gru_roc_auc"] \
        else "does not widen"
    print(f"\n[sweep] Transformer − GRU gap {trend} from seq_len {lengths[0]} to {lengths[-1]} "
          f"({rows[0]['ts_minus_gru_roc_auc']:+.4f} → {rows[-1]['ts_minus_gru_roc_auc']:+.4f})")

    out = resolve(base["paths"]["reports"]) / "ts_sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"lengths": lengths, "rows": rows}, indent=2))
    print(f"[sweep] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
