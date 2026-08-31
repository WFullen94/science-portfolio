"""Capstone stage 1 — Mapping: incident text -> candidate ATT&CK technique(s).

Retrieval over the SAME real detection grounding `stix.py` already extracts (697
techniques) — deliberately not the P6 classifier, which only covers its 20 trained
labels. Embedding retrieval generalizes to every technique the generator can produce a
faithful pack for, and it naturally returns **top-k with a score** instead of a forced
single label. That matters: mis-mapping an incident to the wrong technique means every
downstream artifact is faithfully wrong, so a low-confidence match should route to a
human, not silently generate a pack. See `incident.confidence_threshold` in config.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from detsynth.config import load_config, resolve


def _load_grounding(cfg):
    from detsynth.stix import build_grounding
    cache = resolve(cfg["stix"]["cache"])
    if cache.exists():
        return [json.loads(l) for l in cache.read_text().splitlines()]
    return list(build_grounding(cfg).values())


def _doc(rec: dict) -> str:
    """The retrieval document: name + description.

    Tried enriching this with analytic detection prose (closer register to CTI
    procedure text) and it made top-1 accuracy WORSE (22.7% -> 17.7% on the P6
    benchmark, see README) — analytic text is about detection mechanics (log
    sources, event IDs), not adversary behavior, so it dilutes the semantic match
    rather than sharpening it. Reverted; name+description is the better retrieval doc.
    """
    return f"{rec['name']}. {rec['description'][:500]}"


@dataclass
class TechniqueIndex:
    records: list[dict]
    embeddings: np.ndarray  # (N, D), L2-normalized
    embedder: object         # any object exposing .encode(texts, normalize_embeddings=True)

    def query(self, text: str, top_k: int = 3) -> list[dict]:
        q = np.asarray(self.embedder.encode([text], normalize_embeddings=True))[0]
        sims = self.embeddings @ q
        order = np.argsort(sims)[::-1][:top_k]
        return [{"technique_id": self.records[i]["technique_id"],
                 "name": self.records[i]["name"],
                 "score": float(sims[i])} for i in order]


def build_index(cfg=None) -> TechniqueIndex:
    from sentence_transformers import SentenceTransformer

    cfg = cfg or load_config()
    # Only techniques with real analytics — those are the ones the generator can
    # ground a genuinely faithful pack on (matches align_data.py's convention).
    records = [r for r in _load_grounding(cfg) if r["analytics"]]
    embedder = SentenceTransformer(cfg["incident"]["embed_model"])
    embeddings = np.asarray(embedder.encode([_doc(r) for r in records],
                                            normalize_embeddings=True,
                                            show_progress_bar=False))
    return TechniqueIndex(records, embeddings, embedder)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--incident", default=None, help="incident text (default: a demo example)")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    cfg = load_config()
    idx = build_index(cfg)
    print(f"[mapping] indexed {len(idx.records)} techniques (with real analytics)")
    text = args.incident or ("We found a PowerShell process spawned by winword.exe that "
                             "queried DNS for a high-entropy domain, then created a "
                             "scheduled task to persist.")
    print(f"[mapping] incident: {text}")
    for c in idx.query(text, top_k=args.top_k):
        print(f"  {c['score']:.3f}  {c['technique_id']:12}  {c['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
