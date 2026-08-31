"""Capstone serving — incident text in, mapped technique candidates and (on
confirmation) an incident-conditioned detection pack out.

Design driven by the mapping eval (see incident_eval.py / README): retrieval mapping
is a WEAK signal (top-1 accuracy 22.7%, and the score gap between correct/incorrect
matches is small and heavily overlapping). A confidence gate that silently
auto-generates would be gambling on that weak signal. So the primary flow requires a
human to confirm the technique before anything is generated:

    POST /map   {text}                     -> top-k candidates (always)
    POST /pack  {text, technique_id}       -> pack, for a HUMAN-CONFIRMED technique
    POST /incident {text, auto_generate}   -> convenience: candidates, and ONLY IF
                                               auto_generate=true a provisional pack
                                               for top-1 — always labeled provisional.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel

from detsynth.config import load_config
from detsynth.generate import _load_grounding, pack_to_markdown
from detsynth.incident_pack import generate_incident_pack
from detsynth.mapping import build_index
from detsynth.modeling import load_model, load_tokenizer, pick_device

app = FastAPI(title="detsynth — incident to detection pack")


class MapRequest(BaseModel):
    text: str
    top_k: int | None = None


class PackRequest(BaseModel):
    text: str
    technique_id: str


class IncidentRequest(BaseModel):
    text: str
    top_k: int | None = None
    auto_generate: bool = False  # explicit opt-in; still returned as "provisional"


@lru_cache(maxsize=1)
def _load():
    cfg = load_config()
    idx = build_index(cfg)
    grounding = {r["technique_id"]: r for r in _load_grounding(cfg)}
    device = pick_device(cfg["model"]["device"])
    tok = load_tokenizer(cfg["model"]["base"])
    model = load_model(cfg["model"]["base"], device, cfg["model"]["dtype"])
    model.eval()
    return cfg, idx, grounding, model, tok


def _pack_response(pack: dict) -> dict:
    return {"pack": pack, "pack_markdown": pack_to_markdown(pack)}


@app.get("/")
def root():
    cfg, idx, *_ = _load()
    return {"service": "detsynth-incident", "n_techniques_indexed": len(idx.records),
            "mapping_top1_accuracy": 0.227,  # from incident_eval.py — see README
            "confidence_threshold": cfg["incident"]["confidence_threshold"],
            "note": "mapping is a weak signal (see /README) — confirm before trusting"}


@app.get("/health")
def health():
    _load()
    return {"status": "ok"}


@app.post("/map")
def map_incident(req: MapRequest):
    cfg, idx, *_ = _load()
    candidates = idx.query(req.text, top_k=req.top_k or cfg["incident"]["top_k"])
    return {"candidates": candidates,
            "note": "retrieval-only mapping (~23% top-1 accuracy on held-out CTI "
                    "procedures) — confirm the technique before generating a pack"}


@app.post("/pack")
def pack(req: PackRequest):
    cfg, idx, grounding, model, tok = _load()
    if req.technique_id not in grounding:
        return {"error": f"unknown technique_id {req.technique_id!r}"}
    p = generate_incident_pack(req.text, req.technique_id, cfg, grounding, model, tok)
    return {"mode": "human_confirmed", **_pack_response(p)}


@app.post("/incident")
def incident(req: IncidentRequest):
    cfg, idx, grounding, model, tok = _load()
    candidates = idx.query(req.text, top_k=req.top_k or cfg["incident"]["top_k"])
    if not candidates:
        return {"mode": "no_match", "candidates": [], "pack": None}

    result = {"candidates": candidates, "pack": None}
    if req.auto_generate and candidates[0]["score"] >= cfg["incident"]["confidence_threshold"]:
        p = generate_incident_pack(req.text, candidates[0]["technique_id"], cfg, grounding, model, tok)
        result.update(mode="auto_generated_PROVISIONAL", **_pack_response(p),
                      note="mapping was NOT human-confirmed — treat this pack as a "
                           "starting hypothesis, not a validated detection")
    else:
        result.update(mode="needs_confirmation",
                      note="pass technique_id to POST /pack once a human has "
                           "confirmed the technique, or set auto_generate=true for a "
                           "provisional (unconfirmed) pack")
    return result
