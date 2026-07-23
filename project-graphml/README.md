# Graph ML — Link Prediction over MITRE ATT&CK

Treats MITRE ATT&CK as the **knowledge graph it actually is** and trains a Graph Neural Network
to answer a real CTI question: **which techniques is a threat group likely to use?**

It's a recommender system, but for adversary TTPs — Netflix predicts movies from a user-movie
graph; this predicts techniques from a group-technique graph. *Same math (link prediction),
security use case (threat-actor profiling).*

```
ATT&CK STIX -> graph (groups, software, techniques + relationships)
   -> GraphSAGE message passing (learn node embeddings)
   -> link prediction: score group–technique pairs
   -> rank a group's likely-but-undocumented techniques
```

## Stack

| Piece | Tool |
|-------|------|
| Graph / GNN | **PyTorch Geometric** (GraphSAGE) — also the project's **PyTorch** showcase |
| Task | link prediction on `group --uses--> technique` edges |
| Eval | ROC-AUC + average precision on held-out edges (sklearn) |

## The graph

| | count |
|---|---|
| Nodes | 1,692 (697 techniques, 174 groups, 821 software) |
| **Target edges** (`group→technique`) | 4,546 |
| Auxiliary edges (software→technique, group→software, sub-technique) | ~13k |

Target edges are what we predict; auxiliary edges are message-passing structure only (a group's
malware reveals techniques), never labels.

## What makes the evaluation honest

- **No leakage:** val/test edges are removed from the message-passing graph — the model can't see
  the answers it's scored on.
- **Typed negatives:** negatives are sampled as `group→technique` *non-edges*, not random node
  pairs. Otherwise the model would score high just by learning "groups connect to techniques"
  (node-type discrimination) instead of *which* technique.

**Result:** test **ROC-AUC 0.908 / AP 0.899** on held-out edges — a real, non-trivial signal
(0.5 = random).

## Demo

```bash
python -m attackgraph.build_graph      # STIX -> graph
python -m attackgraph.train            # train the GNN (test AUC ~0.91)
python -m attackgraph.predict "APT29"  # rank likely-but-undocumented techniques
```

For **APT29**, the top predictions are Reconnaissance / Resource-Development techniques
(Email Addresses, Domains, VPS, Social Media Accounts, Gather Victim Identity Info) — exactly the
infrastructure-building profile of a sophisticated espionage actor.

## Layout

```
conf/config.yaml       graph split, model, training config
src/attackgraph/
  build_graph.py       STIX -> graph tensors            [stage 1]
  model.py             GraphSAGE encoder + link decoder  [stage 2]
  train.py             link-prediction training + eval   [stage 2]
  predict.py           rank a group's likely techniques  [stage 3]
```

## Extensions
Text-embedding node features (reuse the P3 sentence-transformer) instead of learnable embeddings;
heterogeneous GNN (RGCN) over typed edges; predicting `mitigation→technique` for defensive gap analysis.
