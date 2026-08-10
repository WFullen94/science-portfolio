"""Phase 2d — the before/after: faithfulness of base vs SFT vs DPO.

For each held-out technique, generate telemetry from light grounding, then score how
faithful it is with an embedding **grounding margin**:

    margin = sim(generated, THIS technique's real analytics)
           - sim(generated, the CONFUSABLE technique's analytics)

A higher/more-positive margin means the output stays on the correct technique's real
detection facts instead of drifting to a plausible wrong one — i.e. less hallucination.
We report the mean margin and the faithful-rate (margin > 0) for base -> SFT -> DPO.
"""

from __future__ import annotations

import json

import numpy as np

from detsynth.align_data import prompt_messages
from detsynth.config import load_config, resolve
from detsynth.modeling import load_model, load_tokenizer, pick_device


def _load_variant(kind: str, cfg, device):
    acfg = cfg["align"]
    base, dt = acfg["base"], acfg["dtype"]
    merged = str(resolve(acfg["sft"]["out_dir"] + "_merged"))
    if kind == "base":
        return load_model(base, device, dt), load_tokenizer(base)
    if kind == "sft":
        return load_model(merged, device, dt), load_tokenizer(merged)
    if kind == "dpo":
        from peft import PeftModel
        m = load_model(merged, device, dt)
        m = PeftModel.from_pretrained(m, str(resolve(acfg["dpo"]["out_dir"]))).to(device)
        return m, load_tokenizer(merged)
    raise ValueError(kind)


def _gen(model, tok, item, device, max_new_tokens) -> str:
    import torch
    enc = tok.apply_chat_template(prompt_messages(item), add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def run(cfg=None) -> dict:
    from sentence_transformers import SentenceTransformer

    cfg = cfg or load_config()
    acfg = cfg["align"]
    device = pick_device(cfg["model"]["device"])
    items = [json.loads(l) for l in
             (resolve(acfg["prefs_dir"]) / "test.jsonl").read_text().splitlines()]

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    gold = embedder.encode([it["gold_telemetry"] for it in items], normalize_embeddings=True)
    conf = embedder.encode([it["confusable_telemetry"] for it in items], normalize_embeddings=True)

    results = {}
    for kind in ("base", "sft", "dpo"):
        model, tok = _load_variant(kind, cfg, device)
        gens = [_gen(model, tok, it, device, acfg["max_new_tokens"]) for it in items]
        G = embedder.encode(gens, normalize_embeddings=True)
        margin = (G * gold).sum(1) - (G * conf).sum(1)
        results[kind] = {"margin_mean": float(margin.mean()),
                         "faithful_rate": float((margin > 0).mean()), "n": len(items)}
        print(f"[eval] {kind:>4}: margin={results[kind]['margin_mean']:+.4f} "
              f"faithful_rate={results[kind]['faithful_rate']:.4f}")
        del model

    print("\n[eval] faithfulness (held-out techniques): grounding margin = "
          "sim(gen, correct) − sim(gen, confusable)\n")
    print("| model | margin (mean) | faithful-rate |")
    print("|---|---|---|")
    for k in ("base", "sft", "dpo"):
        print(f"| {k.upper():<4} | {results[k]['margin_mean']:+.4f} | {results[k]['faithful_rate']:.4f} |")
    print(f"\n[eval] DPO − SFT margin Δ = {results['dpo']['margin_mean'] - results['sft']['margin_mean']:+.4f}")

    out = resolve("data/reports") / "faithfulness.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    try:
        import mlflow
        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        mlflow.set_experiment(cfg["mlflow"]["experiment"])
        with mlflow.start_run(run_name="faithfulness-base-sft-dpo"):
            for k, v in results.items():
                mlflow.log_metric(f"{k}_margin", v["margin_mean"])
                mlflow.log_metric(f"{k}_faithful_rate", v["faithful_rate"])
    except Exception as exc:
        print(f"[eval] mlflow skipped: {exc}")
    print(f"[eval] wrote {out}")
    return results


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
