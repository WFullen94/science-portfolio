"""Stage 1 — Ingest: MITRE ATT&CK STIX bundle -> technique documents.

Downloads the official ATT&CK Enterprise STIX 2.1 bundle, parses the active
attack-pattern (technique) objects, and writes one retrieval-ready document per
technique to documents.jsonl. Each document composes the technique id, name,
tactics, platforms, and description into a single searchable text field, with
metadata kept for citation and filtering.
"""

from __future__ import annotations

import json

import requests

from rag.config import load_config, resolve


def download_stix(cfg) -> dict:
    cache = resolve(cfg["corpus"]["cache"])
    if cache.exists():
        print(f"[ingest] using cached STIX bundle {cache}")
        return json.loads(cache.read_text())
    cache.parent.mkdir(parents=True, exist_ok=True)
    url = cfg["corpus"]["stix_url"]
    print(f"[ingest] downloading {url}")
    data = requests.get(url, timeout=180).json()
    cache.write_text(json.dumps(data))
    return data


def _mitre_ref(obj: dict) -> dict:
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return ref
    return {}


def _compose_text(tid, name, tactics, platforms, is_sub, description) -> str:
    header = f"MITRE ATT&CK {'Sub-technique' if is_sub else 'Technique'} {tid}: {name}"
    meta = []
    if tactics:
        meta.append(f"Tactics: {', '.join(tactics)}")
    if platforms:
        meta.append(f"Platforms: {', '.join(platforms)}")
    return f"{header}\n" + ("\n".join(meta) + "\n\n" if meta else "\n") + description


def parse_techniques(data: dict) -> list[dict]:
    docs = []
    for obj in data["objects"]:
        if obj.get("type") != "attack-pattern":
            continue
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue
        ref = _mitre_ref(obj)
        tid = ref.get("external_id")
        if not tid:
            continue
        tactics = [p["phase_name"] for p in obj.get("kill_chain_phases", [])]
        platforms = obj.get("x_mitre_platforms", [])
        is_sub = bool(obj.get("x_mitre_is_subtechnique"))
        name = obj["name"]
        docs.append({
            "id": tid,
            "technique_id": tid,
            "name": name,
            "tactics": tactics,
            "platforms": platforms,
            "is_subtechnique": is_sub,
            "url": ref.get("url", ""),
            "text": _compose_text(tid, name, tactics, platforms, is_sub, obj["description"]),
        })
    return docs


def main() -> int:
    cfg = load_config()
    data = download_stix(cfg)
    docs = parse_techniques(data)

    out = resolve(cfg["paths"]["documents"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for d in docs:
            fh.write(json.dumps(d) + "\n")

    n_sub = sum(d["is_subtechnique"] for d in docs)
    print(f"[ingest] wrote {len(docs)} technique docs "
          f"({len(docs) - n_sub} techniques + {n_sub} sub-techniques) -> {out}")
    print(f"[ingest] sample: {docs[0]['technique_id']} — {docs[0]['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
