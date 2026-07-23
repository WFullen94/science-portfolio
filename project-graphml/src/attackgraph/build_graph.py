"""Stage 1 — Build the ATT&CK knowledge graph from the STIX bundle.

Nodes: techniques (attack-pattern), threat groups (intrusion-set), software
(malware/tool). Edges come from MITRE's published relationships:

  target (what we predict):  group  --uses-->      technique
  auxiliary (structure):     software --uses-->     technique
                             group    --uses-->     software
                             technique --subtech-of--> technique

The auxiliary edges enrich message passing (a group's malware reveals techniques)
but are never used as prediction labels. Saves a small dict of tensors + name maps.
"""

from __future__ import annotations

import json

import requests
import torch

from attackgraph.config import load_config, resolve

# Node type codes.
TECHNIQUE, GROUP, SOFTWARE = 0, 1, 2
TYPE_NAME = {TECHNIQUE: "technique", GROUP: "group", SOFTWARE: "software"}

_STIX_TYPE = {
    "attack-pattern": TECHNIQUE,
    "intrusion-set": GROUP,
    "malware": SOFTWARE,
    "tool": SOFTWARE,
}


def download_stix(cfg) -> dict:
    cache = resolve(cfg["corpus"]["cache"])
    if cache.exists():
        print(f"[graph] using cached STIX {cache}")
        return json.loads(cache.read_text())
    cache.parent.mkdir(parents=True, exist_ok=True)
    print(f"[graph] downloading {cfg['corpus']['stix_url']}")
    data = requests.get(cfg["corpus"]["stix_url"], timeout=180).json()
    cache.write_text(json.dumps(data))
    return data


def _active(obj: dict) -> bool:
    return not obj.get("revoked") and not obj.get("x_mitre_deprecated")


def _ext_id(obj: dict) -> str:
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return ref.get("external_id", "")
    return ""


def build_graph(data: dict) -> dict:
    # --- nodes ---
    stix_to_idx: dict[str, int] = {}
    node_type, names, ext_ids = [], [], []
    for obj in data["objects"]:
        t = _STIX_TYPE.get(obj.get("type"))
        if t is None or not _active(obj):
            continue
        stix_to_idx[obj["id"]] = len(node_type)
        node_type.append(t)
        names.append(obj.get("name", "?"))
        ext_ids.append(_ext_id(obj) if t == TECHNIQUE else "")

    def kind(stix_id: str):
        return _STIX_TYPE.get(stix_id.split("--", 1)[0])

    # --- edges ---
    target, sw_tech, grp_sw, subtech = [], [], [], []
    for obj in data["objects"]:
        if obj.get("type") != "relationship" or not _active(obj):
            continue
        rt = obj.get("relationship_type")
        s, d = obj.get("source_ref", ""), obj.get("target_ref", "")
        if s not in stix_to_idx or d not in stix_to_idx:
            continue
        si, di = stix_to_idx[s], stix_to_idx[d]
        sk, dk = kind(s), kind(d)
        if rt == "uses" and sk == GROUP and dk == TECHNIQUE:
            target.append((si, di))
        elif rt == "uses" and sk == SOFTWARE and dk == TECHNIQUE:
            sw_tech.append((si, di))
        elif rt == "uses" and sk == GROUP and dk == SOFTWARE:
            grp_sw.append((si, di))
        elif rt == "subtechnique-of" and sk == TECHNIQUE and dk == TECHNIQUE:
            subtech.append((si, di))

    def to_edge_index(pairs):
        if not pairs:
            return torch.empty(2, 0, dtype=torch.long)
        return torch.tensor(pairs, dtype=torch.long).t().contiguous()

    aux = sw_tech + grp_sw + subtech
    return {
        "num_nodes": len(node_type),
        "node_type": torch.tensor(node_type, dtype=torch.long),
        "names": names,
        "ext_ids": ext_ids,
        "stix_to_idx": stix_to_idx,
        "target_edges": to_edge_index(target),      # group -> technique
        "aux_edges": to_edge_index(aux),             # software/sub-technique structure
        "counts": {
            "technique": node_type.count(TECHNIQUE),
            "group": node_type.count(GROUP),
            "software": node_type.count(SOFTWARE),
            "group_uses_technique": len(target),
            "software_uses_technique": len(sw_tech),
            "group_uses_software": len(grp_sw),
            "subtechnique_of": len(subtech),
        },
    }


def main() -> int:
    cfg = load_config()
    graph = build_graph(download_stix(cfg))
    out = resolve(cfg["paths"]["graph"])
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph, out)

    c = graph["counts"]
    print(f"[graph] nodes: {graph['num_nodes']} "
          f"({c['technique']} techniques, {c['group']} groups, {c['software']} software)")
    print(f"[graph] TARGET group->technique edges: {c['group_uses_technique']}")
    print(f"[graph] aux edges: {c['software_uses_technique']} sw->tech, "
          f"{c['group_uses_software']} grp->sw, {c['subtechnique_of']} sub-technique")
    print(f"[graph] saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
