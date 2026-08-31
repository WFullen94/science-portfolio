"""Capstone stage 2 — Generate a detection pack conditioned on an incident.

Phase 1's `generate.py` produces a pack from a technique's grounding alone — the same
generic pack every time for that technique. This conditions generation on BOTH the
technique's real ATT&CK grounding AND the incident's own specifics (the hosts,
processes, tools actually mentioned), so the pack responds to what was observed
instead of reading like technique-page boilerplate.

The anti-hallucination instruction extends the same way Phase 1 does: don't invent
platforms/log sources/event IDs beyond the grounding, and don't invent entities beyond
what the incident mentions.
"""

from __future__ import annotations

import argparse
import json

from detsynth.config import load_config, resolve
from detsynth.generate import INSTRUCTIONS, _gen, _grounding_block, _load_grounding, pack_to_markdown
from detsynth.modeling import load_model, load_tokenizer, pick_device

INCIDENT_SYSTEM = (
    "You are a senior detection engineer triaging a real incident. Using ONLY the "
    "MITRE ATT&CK grounding and the incident details provided, produce the requested "
    "detection artifact. Be specific and faithful: do not invent platforms, log "
    "sources, or event IDs the grounding does not support, and do not invent hosts, "
    "users, or tools beyond what the incident mentions. Reference the incident's own "
    "specifics where they inform the artifact. Output the artifact only — no preamble."
)

INCIDENT_SUFFIX = (" Ground it in the incident's specific details (hosts, processes, "
                   "tools) where they matter — don't just restate the technique generically.")


def _incident_block(incident_text: str, rec: dict, max_analytics: int) -> str:
    return (f"INCIDENT:\n{incident_text.strip()}\n\n"
            f"MAPPED TECHNIQUE GROUNDING:\n{_grounding_block(rec, max_analytics)}")


def generate_incident_pack(incident_text: str, technique_id: str, cfg=None,
                           grounding=None, model=None, tok=None) -> dict:
    cfg = cfg or load_config()
    gcfg = cfg["generate"]
    grounding = grounding or {r["technique_id"]: r for r in _load_grounding(cfg)}
    rec = grounding[technique_id]
    device = pick_device(cfg["model"]["device"])
    if model is None:
        tok = load_tokenizer(cfg["model"]["base"])
        model = load_model(cfg["model"]["base"], device, cfg["model"]["dtype"])
        model.eval()

    ctx = _incident_block(incident_text, rec, gcfg["max_analytics"])
    pack = {"technique_id": technique_id, "name": rec["name"], "platforms": rec["platforms"],
            "grounded_on_analytics": len(rec["analytics"]),
            "incident_excerpt": incident_text.strip()[:300], "artifacts": {}}
    for art in gcfg["artifacts"]:
        pack["artifacts"][art] = _gen(model, tok, device, ctx, INSTRUCTIONS[art] + INCIDENT_SUFFIX,
                                      cfg["model"]["max_new_tokens"], cfg["model"]["temperature"],
                                      system=INCIDENT_SYSTEM)
        print(f"[incident-gen] {technique_id} :: {art} ({len(pack['artifacts'][art])} chars)")
    return pack


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a pack for a HUMAN-CONFIRMED "
                                 "technique_id (see mapping.py to find candidates).")
    ap.add_argument("--technique-id", required=True)
    ap.add_argument("--incident", required=True)
    args = ap.parse_args()

    cfg = load_config()
    pack = generate_incident_pack(args.incident, args.technique_id, cfg)
    out = resolve(cfg["generate"]["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    stem = f"incident-{args.technique_id}"
    (out / f"{stem}.json").write_text(json.dumps(pack, indent=2))
    (out / f"{stem}.md").write_text(pack_to_markdown(pack))
    print(f"[incident-gen] wrote {out / (stem + '.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
