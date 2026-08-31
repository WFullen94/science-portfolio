"""Phase 1b — Generate a grounded 4-part detection pack for a technique.

Every artifact is generated from the technique's real ATT&CK grounding (description,
platforms, and MITRE's own analytics with their tunable knobs), with an explicit
instruction not to invent log sources or platforms beyond it. That grounding is what
keeps output faithful — a starting point a detection engineer validates, not fiction.

  procedures  — concrete ways the technique is carried out
  telemetry   — log/event sources, fields, command-lines, process chains to watch
  sigma       — a Sigma rule skeleton grounded in that telemetry
  fixtures    — positives that should fire, a hard-negative, and a benign case

PoC: a small local model. Swap `model.base` for a larger/API model for real use.
"""

from __future__ import annotations

import argparse
import json

from detsynth.config import load_config, resolve
from detsynth.modeling import load_model, load_tokenizer, pick_device
from detsynth.stix import build_grounding

SYSTEM = (
    "You are a senior detection engineer. Using ONLY the MITRE ATT&CK grounding "
    "provided, produce the requested detection artifact. Be specific and faithful: "
    "do not invent platforms, log sources, or event IDs that the grounding does not "
    "support. Output the artifact only — no preamble."
)

INSTRUCTIONS = {
    "procedures": "Write 3 concise, concrete example adversary procedures for this "
                  "technique — how an actor actually carries it out. One sentence each.",
    "telemetry": "List the detection telemetry to watch: log/event sources, key fields, "
                 "example command-lines or event IDs, and parent/child process chains. "
                 "Ground every item in the analytics above.",
    "sigma": "Write a Sigma rule (valid YAML) to detect this technique, grounded in the "
             "telemetry. Include title, status, logsource, detection, condition, "
             "falsepositives, level, and tags (attack.<tactic>, attack.<technique_id>).",
    "fixtures": "Provide test fixtures: 2 POSITIVE examples (should fire the detection), "
                "1 HARD-NEGATIVE from a confusable technique (name it), and 1 BENIGN "
                "example (must not fire). Label each line.",
}


def _grounding_block(rec: dict, max_analytics: int) -> str:
    lines = [
        f"Technique: {rec['technique_id']} — {rec['name']}",
        f"Platforms: {', '.join(rec['platforms']) or 'n/a'}",
        f"Tactics: {', '.join(rec['tactics']) or 'n/a'}",
        f"Description: {rec['description'][:600]}",
    ]
    if rec["analytics"]:
        lines.append("MITRE detection analytics:")
        for a in rec["analytics"][:max_analytics]:
            knobs = f" [tunable: {', '.join(a['mutable_elements'])}]" if a["mutable_elements"] else ""
            lines.append(f"  - {a['detection'][:400]}{knobs}")
    return "\n".join(lines)


def _gen(model, tok, device, grounding: str, instruction: str, max_new_tokens: int, temp: float,
        system: str = SYSTEM) -> str:
    import torch
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": f"{grounding}\n\nTASK: {instruction}"}]
    enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=temp > 0,
                             temperature=temp or None, top_p=0.9 if temp > 0 else None,
                             pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def generate_pack(technique_id: str, cfg=None, grounding=None, model=None, tok=None) -> dict:
    cfg = cfg or load_config()
    gcfg = cfg["generate"]
    grounding = grounding or {r["technique_id"]: r for r in _load_grounding(cfg)}
    rec = grounding[technique_id]
    device = pick_device(cfg["model"]["device"])
    if model is None:
        tok = load_tokenizer(cfg["model"]["base"])
        model = load_model(cfg["model"]["base"], device, cfg["model"]["dtype"])
        model.eval()

    ctx = _grounding_block(rec, gcfg["max_analytics"])
    pack = {"technique_id": technique_id, "name": rec["name"],
            "platforms": rec["platforms"], "grounded_on_analytics": len(rec["analytics"]),
            "artifacts": {}}
    for art in gcfg["artifacts"]:
        pack["artifacts"][art] = _gen(model, tok, device, ctx, INSTRUCTIONS[art],
                                      cfg["model"]["max_new_tokens"], cfg["model"]["temperature"])
        print(f"[gen] {technique_id} :: {art} ({len(pack['artifacts'][art])} chars)")
    return pack


def _load_grounding(cfg):
    cache = resolve(cfg["stix"]["cache"])
    if cache.exists():
        return [json.loads(l) for l in cache.read_text().splitlines()]
    return list(build_grounding(cfg).values())


def pack_to_markdown(pack: dict) -> str:
    md = [f"# Detection pack — {pack['technique_id']} {pack['name']}",
          f"*Platforms: {', '.join(pack['platforms'])} · grounded on "
          f"{pack['grounded_on_analytics']} MITRE analytic(s)*"]
    if pack.get("incident_excerpt"):
        md.append(f"\n> **Incident:** {pack['incident_excerpt']}")
    md.append("")
    titles = {"procedures": "Example procedures", "telemetry": "Detection telemetry",
              "sigma": "Sigma rule (starter)", "fixtures": "Test fixtures"}
    for art, body in pack["artifacts"].items():
        fence = "\n```yaml\n" + body + "\n```" if art == "sigma" else "\n" + body
        md.append(f"## {titles.get(art, art)}{fence}\n")
    return "\n".join(md)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("technique_ids", nargs="+", help="e.g. T1053.005 T1059.001")
    args = ap.parse_args()
    cfg = load_config()
    out = resolve(cfg["generate"]["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    grounding = {r["technique_id"]: r for r in _load_grounding(cfg)}
    device = pick_device(cfg["model"]["device"])
    tok = load_tokenizer(cfg["model"]["base"])
    model = load_model(cfg["model"]["base"], device, cfg["model"]["dtype"])
    model.eval()
    for tid in args.technique_ids:
        pack = generate_pack(tid, cfg, grounding, model, tok)
        (out / f"{tid}.json").write_text(json.dumps(pack, indent=2))
        (out / f"{tid}.md").write_text(pack_to_markdown(pack))
        print(f"[gen] wrote {out / (tid + '.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
