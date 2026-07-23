"""Stage 3 — Predict a threat group's likely techniques (the CTI payoff).

Loads the trained GNN, encodes the graph once, then for a given group scores every
technique by the dot product of their learned embeddings. Techniques the group is
already documented using are marked [known]; the high-scoring ones it is NOT yet
documented using are the model's [NEW] predictions — candidate TTPs to hunt for.
"""

from __future__ import annotations

import sys

import torch

from attackgraph.build_graph import GROUP, TECHNIQUE
from attackgraph.config import load_config, resolve
from attackgraph.model import LinkGNN


def load_encoded(cfg):
    graph = torch.load(resolve(cfg["paths"]["graph"]), weights_only=False)
    ckpt = torch.load(resolve(cfg["paths"]["model"]), weights_only=False)
    model = LinkGNN(ckpt["num_nodes"], ckpt["embed_dim"], ckpt["hidden_dim"],
                    ckpt["num_layers"], ckpt["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with torch.no_grad():
        z = model.encode(ckpt["message_edge_index"])
    return graph, z


def find_group(graph, query: str):
    q = query.lower()
    matches = [
        (i, n) for i, (t, n) in enumerate(zip(graph["node_type"].tolist(), graph["names"]))
        if t == GROUP and q in n.lower()
    ]
    return matches


def rank_techniques(graph, z, group_idx: int):
    tech_ids = torch.where(graph["node_type"] == TECHNIQUE)[0]
    scores = torch.sigmoid((z[group_idx] * z[tech_ids]).sum(dim=-1))
    known = {int(d) for s, d in graph["target_edges"].t().tolist() if int(s) == group_idx}
    order = torch.argsort(scores, descending=True)
    ranked = []
    for j in order.tolist():
        tid = int(tech_ids[j])
        ranked.append({
            "technique_id": graph["ext_ids"][tid],
            "name": graph["names"][tid],
            "score": round(float(scores[j]), 4),
            "known": tid in known,
        })
    return ranked


def main() -> int:
    cfg = load_config()
    query = " ".join(sys.argv[1:]) or "APT29"
    graph, z = load_encoded(cfg)
    matches = find_group(graph, query)
    if not matches:
        print(f"No threat group matching '{query}'. Try e.g. APT29, Lazarus, FIN7.")
        return 1
    group_idx, group_name = matches[0]
    ranked = rank_techniques(graph, z, group_idx)

    print(f"Threat group: {group_name}")
    n_known = sum(r["known"] for r in ranked)
    top_known = [r for r in ranked if r["known"]][:200]
    recovered = sum(1 for r in ranked[:n_known] if r["known"])
    print(f"  documented techniques: {n_known}; "
          f"{recovered} of them fall in the model's top {n_known} (sanity check)\n")

    print("  Top NEW predicted techniques (not yet documented for this group):")
    shown = 0
    for r in ranked:
        if not r["known"]:
            print(f"    {r['score']:.3f}  {r['technique_id']:10s} {r['name']}")
            shown += 1
            if shown == 10:
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
