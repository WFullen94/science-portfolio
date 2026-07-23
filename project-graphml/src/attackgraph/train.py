"""Stage 2 — Train the GNN for link prediction of group->technique edges.

Splits the target edges into train/val/test. The message-passing graph uses only
TRAIN target edges (+ auxiliary structure) so val/test edges are never seen — no
leakage. Negatives are sampled as group->technique NON-edges (not random node
pairs), so the model must learn *which* technique a group uses, not merely that
groups connect to techniques. Reports ROC-AUC and average precision.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from attackgraph.build_graph import GROUP, TECHNIQUE
from attackgraph.config import load_config, resolve
from attackgraph.model import LinkGNN


def _undirected(edge_index: torch.Tensor) -> torch.Tensor:
    return torch.cat([edge_index, edge_index.flip(0)], dim=1)


def _split(edge_index, val_frac, test_frac, gen):
    n = edge_index.size(1)
    perm = torch.randperm(n, generator=gen)
    n_val, n_test = int(n * val_frac), int(n * test_frac)
    test_i, val_i, train_i = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]
    return edge_index[:, train_i], edge_index[:, val_i], edge_index[:, test_i]


def _sample_negatives(n, group_ids, tech_ids, existing, gen):
    """Sample n group->technique pairs that are NOT real edges."""
    g_idx = torch.randint(len(group_ids), (n * 2,), generator=gen)
    t_idx = torch.randint(len(tech_ids), (n * 2,), generator=gen)
    src, dst = [], []
    for gi, ti in zip(g_idx.tolist(), t_idx.tolist()):
        g, t = int(group_ids[gi]), int(tech_ids[ti])
        if (g, t) not in existing:
            src.append(g); dst.append(t)
        if len(src) == n:
            break
    return torch.tensor([src, dst], dtype=torch.long)


@torch.no_grad()
def _evaluate(model, msg_ei, pos_ei, neg_ei) -> tuple[float, float]:
    model.eval()
    z = model.encode(msg_ei)
    scores = torch.cat([model.decode(z, pos_ei), model.decode(z, neg_ei)]).sigmoid().numpy()
    labels = np.concatenate([np.ones(pos_ei.size(1)), np.zeros(neg_ei.size(1))])
    return roc_auc_score(labels, scores), average_precision_score(labels, scores)


def main() -> int:
    cfg = load_config()
    gcfg, mcfg, tcfg = cfg["graph"], cfg["model"], cfg["train"]
    torch.manual_seed(tcfg["seed"])
    gen = torch.Generator().manual_seed(tcfg["seed"])

    graph = torch.load(resolve(cfg["paths"]["graph"]), weights_only=False)
    num_nodes = graph["num_nodes"]
    node_type = graph["node_type"]
    group_ids = torch.where(node_type == GROUP)[0]
    tech_ids = torch.where(node_type == TECHNIQUE)[0]

    target = graph["target_edges"]
    existing = {(int(s), int(d)) for s, d in target.t().tolist()}
    train_pos, val_pos, test_pos = _split(target, gcfg["val_frac"], gcfg["test_frac"], gen)

    # Message-passing graph: TRAIN target edges + auxiliary structure, undirected.
    msg_ei = _undirected(torch.cat([train_pos, graph["aux_edges"]], dim=1))

    val_neg = _sample_negatives(val_pos.size(1), group_ids, tech_ids, existing, gen)
    test_neg = _sample_negatives(test_pos.size(1), group_ids, tech_ids, existing, gen)

    model = LinkGNN(num_nodes, mcfg["embed_dim"], mcfg["hidden_dim"],
                    mcfg["num_layers"], mcfg["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    loss_fn = torch.nn.BCEWithLogitsLoss()

    print(f"[train] {train_pos.size(1)} train / {val_pos.size(1)} val / "
          f"{test_pos.size(1)} test target edges")
    best_val = 0.0
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        opt.zero_grad()
        z = model.encode(msg_ei)
        train_neg = _sample_negatives(train_pos.size(1), group_ids, tech_ids, existing, gen)
        pos_s, neg_s = model.decode(z, train_pos), model.decode(z, train_neg)
        scores = torch.cat([pos_s, neg_s])
        labels = torch.cat([torch.ones_like(pos_s), torch.zeros_like(neg_s)])
        loss = loss_fn(scores, labels)
        loss.backward()
        opt.step()
        if epoch % 20 == 0 or epoch == 1:
            val_auc, val_ap = _evaluate(model, msg_ei, val_pos, val_neg)
            best_val = max(best_val, val_auc)
            print(f"[train] epoch {epoch:3d}  loss {loss.item():.4f}  "
                  f"val ROC-AUC {val_auc:.4f}  val AP {val_ap:.4f}")

    test_auc, test_ap = _evaluate(model, msg_ei, test_pos, test_neg)
    print(f"[train] TEST ROC-AUC {test_auc:.4f}  AP {test_ap:.4f}")

    torch.save({
        "state_dict": model.state_dict(),
        "num_nodes": num_nodes,
        "embed_dim": mcfg["embed_dim"], "hidden_dim": mcfg["hidden_dim"],
        "num_layers": mcfg["num_layers"], "dropout": mcfg["dropout"],
        "message_edge_index": msg_ei,
    }, resolve(cfg["paths"]["model"]))

    reports = resolve(cfg["paths"]["reports"]); reports.mkdir(parents=True, exist_ok=True)
    (reports / "link_metrics.json").write_text(json.dumps({
        "test_roc_auc": round(test_auc, 4), "test_ap": round(test_ap, 4),
        "best_val_roc_auc": round(best_val, 4),
        "n_train": train_pos.size(1), "n_val": val_pos.size(1), "n_test": test_pos.size(1),
    }, indent=2))
    print(f"[train] saved model + metrics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
