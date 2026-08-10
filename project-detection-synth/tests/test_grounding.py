"""Grounding + formatting tests — pure (no model, no network, no STIX download).

Validates the STIX chain-walk (analytic <- detection-strategy -> technique) on a tiny
synthetic bundle, and the pack markdown rendering. Model/generation paths are verified
locally (they need torch + the model weights).
"""

from __future__ import annotations

import copy
import json

from detsynth.config import load_config
from detsynth.generate import pack_to_markdown
from detsynth.stix import build_grounding

MINI_BUNDLE = {"objects": [
    {"type": "attack-pattern", "id": "attack-pattern--t1",
     "name": "Test Technique", "description": "Adversaries do a thing on Windows.",
     "x_mitre_platforms": ["Windows"], "x_mitre_is_subtechnique": False,
     "kill_chain_phases": [{"phase_name": "execution"}],
     "external_references": [{"source_name": "mitre-attack", "external_id": "T9999"}]},
    {"type": "x-mitre-analytic", "id": "x-mitre-analytic--a1",
     "name": "Analytic A", "description": "Monitor Windows Event 4698 for scheduled tasks.",
     "x_mitre_platforms": ["Windows"],
     "x_mitre_mutable_elements": [{"field": "TimeWindow"}, {"field": "UserContext"}]},
    {"type": "x-mitre-detection-strategy", "id": "x-mitre-detection-strategy--s1",
     "name": "Strategy 1", "x_mitre_analytic_refs": ["x-mitre-analytic--a1"]},
    {"type": "relationship", "id": "relationship--r1", "relationship_type": "detects",
     "source_ref": "x-mitre-detection-strategy--s1", "target_ref": "attack-pattern--t1"},
    # A deprecated technique that must be excluded.
    {"type": "attack-pattern", "id": "attack-pattern--dead", "name": "Dead",
     "x_mitre_deprecated": True,
     "external_references": [{"source_name": "mitre-attack", "external_id": "T0000"}]},
]}


def _cfg_with_bundle(tmp_path):
    p = tmp_path / "mini.json"
    p.write_text(json.dumps(MINI_BUNDLE))
    c = copy.deepcopy(load_config())
    c["stix"]["bundle"] = str(p)
    return c


def test_grounding_walks_analytic_chain(tmp_path):
    g = build_grounding(_cfg_with_bundle(tmp_path))
    assert "T9999" in g and "T0000" not in g          # deprecated excluded
    rec = g["T9999"]
    assert rec["platforms"] == ["Windows"]
    assert rec["tactics"] == ["execution"]
    assert len(rec["analytics"]) == 1
    a = rec["analytics"][0]
    assert "4698" in a["detection"]                    # real analytic prose attached
    assert a["mutable_elements"] == ["TimeWindow", "UserContext"]


def test_pack_markdown_renders_all_artifacts():
    pack = {"technique_id": "T9999", "name": "Test", "platforms": ["Windows"],
            "grounded_on_analytics": 1,
            "artifacts": {"procedures": "p", "telemetry": "t",
                          "sigma": "title: x", "fixtures": "f"}}
    md = pack_to_markdown(pack)
    assert "# Detection pack — T9999 Test" in md
    assert "```yaml\ntitle: x\n```" in md               # sigma fenced as yaml
    for section in ("Example procedures", "Detection telemetry", "Test fixtures"):
        assert section in md
