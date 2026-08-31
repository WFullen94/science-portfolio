"""Capstone eval — mapping accuracy: does retrieval surface the right technique?

Uses P6's held-out CTI procedure test split (short adversary-behavior text + gold
ATT&CK label) as incident-like eval data — same task shape ("short text -> technique")
as an incident, with existing gold labels and zero new labeling needed. Note this set
only covers P6's original 20-technique subset (not the full 697 the index carries), so
it's a reasonable proxy for mapping quality, not exhaustive coverage.

Also reports the score distribution for correct vs. incorrect top-1 matches — that's
what `incident.confidence_threshold` should be calibrated against, not guessed.
"""

from __future__ import annotations

import csv
import json

import numpy as np

from detsynth.config import load_config, resolve
from detsynth.mapping import build_index


def _load_p6_test(cfg, max_items: int | None = None) -> list[dict]:
    path = resolve(cfg["incident"]["p6_csv"])
    with open(path) as fh:
        rows = [r for r in csv.DictReader(fh) if r["split"] == "test"]
    if max_items:
        rows = rows[:max_items]
    return rows


def run(cfg=None) -> dict:
    cfg = cfg or load_config()
    icfg = cfg["incident"]
    idx = build_index(cfg)
    rows = _load_p6_test(cfg, icfg["eval_max_items"])
    top_k = icfg["top_k"]

    correct_scores, incorrect_scores = [], []
    top1 = topk = 0
    for r in rows:
        cands = idx.query(r["text"], top_k=top_k)
        ids = [c["technique_id"] for c in cands]
        is_top1 = bool(ids) and ids[0] == r["technique_id"]
        top1 += is_top1
        topk += r["technique_id"] in ids
        (correct_scores if is_top1 else incorrect_scores).append(cands[0]["score"] if cands else 0.0)

    n = len(rows)
    result = {
        "n": n, "top_1_accuracy": top1 / n, f"top_{top_k}_accuracy": topk / n,
        "top1_score_mean_when_correct": float(np.mean(correct_scores)) if correct_scores else None,
        "top1_score_mean_when_wrong": float(np.mean(incorrect_scores)) if incorrect_scores else None,
        "current_confidence_threshold": icfg["confidence_threshold"],
    }
    print(f"[incident-eval] indexed {len(idx.records)} techniques; "
          f"evaluating on {n} held-out CTI procedures (P6 test split)")
    print(f"[incident-eval] top-1 accuracy={result['top_1_accuracy']:.4f}  "
          f"top-{top_k} accuracy={result[f'top_{top_k}_accuracy']:.4f}")
    print(f"[incident-eval] top-1 score when correct: {result['top1_score_mean_when_correct']:.4f} "
          f"| when wrong: {result['top1_score_mean_when_wrong']:.4f}  "
          f"(gap is what confidence_threshold exploits)")

    out = resolve("data/reports") / "mapping_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"[incident-eval] wrote {out}")
    return result


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
