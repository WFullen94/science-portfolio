"""Phase 2a — Build faithfulness preference data from the ATT&CK grounding.

The failure we saw: even with grounding in the prompt, a small model hallucinates
telemetry for a *different* (confusable) technique — wrong platform, wrong event.
So we teach faithfulness with preference pairs:

  prompt   = light grounding (technique id/name/description/platforms) + "give telemetry"
  chosen   = telemetry built from THIS technique's real MITRE analytics
  rejected = telemetry built from a CONFUSABLE technique's real analytics

The confusable technique is the nearest *other* technique by description embedding —
a genuinely plausible wrong answer. DPO then raises the correct telemetry over the
confusable one. Same split feeds SFT (prompt->chosen) and the held-out faithfulness eval.
"""

from __future__ import annotations

import json

import numpy as np

from detsynth.config import load_config, resolve
from detsynth.stix import build_grounding

SYSTEM = ("You are a detection engineer. Given a MITRE ATT&CK technique, list the "
          "detection telemetry (log/event sources, key fields, command-lines, process "
          "chains) specific to THAT technique. Be faithful to the technique's real "
          "platforms and behavior; do not describe a different technique.")


def prompt_messages(rec: dict) -> list[dict]:
    user = (f"Technique: {rec['technique_id']} — {rec['name']}\n"
            f"Platforms: {', '.join(rec['platforms']) or 'n/a'}\n"
            f"Description: {rec['description'][:400]}\n\n"
            f"List the detection telemetry specific to this technique.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def telemetry_answer(rec: dict, max_analytics: int = 2) -> str:
    lines = [f"Detection telemetry for {rec['name']} ({rec['technique_id']}) — "
             f"platforms: {', '.join(rec['platforms']) or 'n/a'}."]
    for a in rec["analytics"][:max_analytics]:
        lines.append(f"- {a['detection'][:400]}")
        if a["mutable_elements"]:
            lines.append(f"  Tunable: {', '.join(a['mutable_elements'])}")
    return "\n".join(lines)


def _confusable(recs: list[dict]) -> dict[str, int]:
    """tid -> index of nearest OTHER technique by description embedding."""
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = [f"{r['name']}. {r['description'][:400]}" for r in recs]
    V = emb.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    sim = V @ V.T
    np.fill_diagonal(sim, -1.0)
    return {recs[i]["technique_id"]: int(np.argmax(sim[i])) for i in range(len(recs))}


def build(cfg=None) -> dict:
    cfg = cfg or load_config()
    acfg = cfg["align"]
    recs = [r for r in build_grounding(cfg).values() if r["analytics"]]
    rng = np.random.default_rng(acfg["seed"])
    idx = rng.permutation(len(recs))
    recs = [recs[i] for i in idx]

    confus = _confusable(recs)
    by_tid = {r["technique_id"]: i for i, r in enumerate(recs)}

    n_test = acfg["n_test"]
    test_recs, train_recs = recs[:n_test], recs[n_test:]
    cap = min(acfg["max_train"], len(train_recs))
    train_recs = train_recs[:cap]

    sft_rows, dpo_rows = [], []
    for r in train_recs:
        neg = recs[confus[r["technique_id"]]]
        msgs = prompt_messages(r)
        sft_rows.append({"messages": msgs + [{"role": "assistant", "content": telemetry_answer(r)}]})
        dpo_rows.append({
            "prompt": msgs,
            "chosen": [{"role": "assistant", "content": telemetry_answer(r)}],
            "rejected": [{"role": "assistant", "content": telemetry_answer(neg)}],
        })

    out = resolve(acfg["prefs_dir"])
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "sft.jsonl", sft_rows)
    _write(out / "dpo.jsonl", dpo_rows)
    # Test: technique + its own analytics + the confusable's analytics (for margin eval).
    test = [{"technique_id": r["technique_id"], "name": r["name"],
             "description": r["description"], "platforms": r["platforms"],
             "gold_telemetry": telemetry_answer(r),
             "confusable_telemetry": telemetry_answer(recs[confus[r["technique_id"]]])}
            for r in test_recs]
    _write(out / "test.jsonl", test)

    print(f"[align-data] SFT={len(sft_rows)} DPO={len(dpo_rows)} test={len(test)} -> {out}")
    ex = train_recs[0]
    print(f"[align-data] example: {ex['technique_id']} {ex['name']} "
          f"| confusable={recs[confus[ex['technique_id']]]['technique_id']} "
          f"{recs[confus[ex['technique_id']]]['name']}")
    return {"sft": len(sft_rows), "dpo": len(dpo_rows), "test": len(test)}


def _write(path, rows):
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def main() -> int:
    build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
