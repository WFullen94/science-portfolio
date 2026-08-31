"""Capstone tests: mapping ranking + incident-conditioned prompt composition + markdown.

Pure (no model download, no network) — TechniqueIndex.query is tested with a fake
embedder so ranking logic is verified without sentence-transformers. Generation itself
(incident_pack.generate_incident_pack) needs torch + model weights and is verified
locally (see README for real mapping-accuracy + example numbers).
"""

from __future__ import annotations

import numpy as np

from detsynth.generate import pack_to_markdown
from detsynth.incident_pack import INCIDENT_SYSTEM, _incident_block
from detsynth.mapping import TechniqueIndex, _doc


class _FakeEmbedder:
    """Deterministic 2D embedding: [1,0] if 'windows' in text, else [0,1]."""

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        return np.array([[1.0, 0.0] if "windows" in t.lower() else [0.0, 1.0] for t in texts])


RECORDS = [
    {"technique_id": "T1", "name": "Windows Thing", "description": "runs on windows"},
    {"technique_id": "T2", "name": "Other Thing", "description": "runs elsewhere"},
]


def test_technique_index_ranks_by_similarity():
    embedder = _FakeEmbedder()
    embeddings = embedder.encode([_doc(r) for r in RECORDS])
    idx = TechniqueIndex(RECORDS, embeddings, embedder)

    top = idx.query("something happened on a windows host", top_k=1)
    assert top[0]["technique_id"] == "T1"
    assert top[0]["score"] > 0.9  # exact match on the fake embedding axis


def test_technique_index_top_k_orders_all_candidates():
    embedder = _FakeEmbedder()
    embeddings = embedder.encode([_doc(r) for r in RECORDS])
    idx = TechniqueIndex(RECORDS, embeddings, embedder)

    results = idx.query("windows event", top_k=2)
    assert [r["technique_id"] for r in results] == ["T1", "T2"]
    assert results[0]["score"] > results[1]["score"]


def test_incident_block_contains_incident_and_grounding():
    rec = {"technique_id": "T9999", "name": "Test", "platforms": ["Windows"],
           "tactics": ["execution"], "description": "Adversaries do a thing.",
           "analytics": []}
    block = _incident_block("PowerShell spawned from winword.exe", rec, max_analytics=4)
    assert "PowerShell spawned from winword.exe" in block
    assert "T9999" in block and "Test" in block
    assert "INCIDENT:" in block and "MAPPED TECHNIQUE GROUNDING:" in block


def test_incident_system_prompt_forbids_inventing_entities():
    assert "do not invent hosts" in INCIDENT_SYSTEM.lower() or "invent hosts" in INCIDENT_SYSTEM


def test_pack_markdown_renders_incident_excerpt_when_present():
    pack = {"technique_id": "T9999", "name": "Test", "platforms": ["Windows"],
            "grounded_on_analytics": 0, "incident_excerpt": "host WIN-01 ran mimikatz",
            "artifacts": {"procedures": "p", "telemetry": "t", "sigma": "x", "fixtures": "f"}}
    md = pack_to_markdown(pack)
    assert "**Incident:** host WIN-01 ran mimikatz" in md


def test_pack_markdown_omits_incident_section_when_absent():
    pack = {"technique_id": "T9999", "name": "Test", "platforms": ["Windows"],
            "grounded_on_analytics": 0,
            "artifacts": {"procedures": "p", "telemetry": "t", "sigma": "x", "fixtures": "f"}}
    md = pack_to_markdown(pack)
    assert "**Incident:**" not in md
