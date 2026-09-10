"""Phase 1a — Grounding: walk the ATT&CK STIX bundle into per-technique detection facts.

The value here is that ATT&CK v17+ ships *real* detection content most people never
see: `x-mitre-analytic` objects with actual monitoring prose, platforms, log sources,
and tunable "mutable elements". They attach to a technique through a detection strategy:

    analytic  <-(x_mitre_analytic_refs)-  detection-strategy  -(detects)->  technique

We invert that chain so each technique carries its real analytics. Everything the
generator produces downstream is grounded in these facts, so faithfulness is checkable
(and hallucination — an invented log source, a wrong platform — is detectable).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from detsynth.config import load_config, resolve


def _ensure_bundle(cfg) -> Path:
    """Download the public ATT&CK STIX bundle if the configured path doesn't exist
    locally (e.g. on a fresh machine that doesn't have P6's copy). The bundle is
    public — no auth needed — so this makes the project self-contained.
    """
    path = resolve(cfg["stix"]["bundle"])
    if path.exists():
        return path
    import requests

    url = cfg["stix"]["download_url"]
    print(f"[stix] {path} not found locally — downloading from {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = requests.get(url, timeout=120).json()
    path.write_text(json.dumps(data))
    print(f"[stix] cached bundle -> {path}")
    return path


def _mitre_id(obj: dict) -> str | None:
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return ref.get("external_id")
    return None


def _live(obj: dict) -> bool:
    return not (obj.get("revoked") or obj.get("x_mitre_deprecated"))


def build_grounding(cfg=None) -> dict[str, dict]:
    cfg = cfg or load_config()
    bundle = json.loads(_ensure_bundle(cfg).read_text())
    objs = bundle["objects"]
    by_id = {o["id"]: o for o in objs if "id" in o}

    # Analytics: the real detection prose + tunable knobs.
    def analytic_view(a: dict) -> dict:
        return {
            "name": a.get("name", ""),
            "detection": (a.get("description") or "").strip(),
            "platforms": a.get("x_mitre_platforms", []),
            "mutable_elements": [m.get("field") for m in a.get("x_mitre_mutable_elements", [])],
        }

    # detection-strategy -> its analytic views
    strat_analytics = {
        o["id"]: [analytic_view(by_id[r]) for r in o.get("x_mitre_analytic_refs", []) if r in by_id]
        for o in objs if o.get("type") == "x-mitre-detection-strategy" and _live(o)
    }

    # technique(stix id) -> [detection-strategy ids]   (from `detects` relationships)
    tech_strats: dict[str, list[str]] = defaultdict(list)
    for o in objs:
        if o.get("type") == "relationship" and o.get("relationship_type") == "detects":
            tech_strats[o["target_ref"]].append(o["source_ref"])

    grounding = {}
    for o in objs:
        if o.get("type") != "attack-pattern" or not _live(o):
            continue
        tid = _mitre_id(o)
        if not tid:
            continue
        analytics = []
        for sid in tech_strats.get(o["id"], []):
            analytics.extend(strat_analytics.get(sid, []))
        grounding[tid] = {
            "technique_id": tid,
            "name": o["name"],
            "is_subtechnique": bool(o.get("x_mitre_is_subtechnique")),
            "description": (o.get("description") or "").strip(),
            "platforms": o.get("x_mitre_platforms", []),
            "tactics": [p["phase_name"] for p in o.get("kill_chain_phases", [])],
            "analytics": analytics,
        }
    return grounding


def main() -> int:
    cfg = load_config()
    g = build_grounding(cfg)
    out = resolve(cfg["stix"]["cache"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for tid, rec in g.items():
            fh.write(json.dumps(rec) + "\n")
    with_an = sum(1 for r in g.values() if r["analytics"])
    n_an = sum(len(r["analytics"]) for r in g.values())
    print(f"[stix] {len(g)} techniques -> {out}")
    print(f"[stix] {with_an} have detection analytics ({n_an} analytics total)")
    ex = next(r for r in g.values() if r["analytics"])
    print(f"[stix] example: {ex['technique_id']} {ex['name']} | "
          f"{len(ex['analytics'])} analytics | platforms={ex['platforms']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
