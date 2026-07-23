"""Model + training-helper unit tests (no full training run)."""

import torch

from attackgraph.model import LinkGNN
from attackgraph.train import _sample_negatives, _split, _undirected


def test_encode_decode_shapes():
    m = LinkGNN(num_nodes=6, embed_dim=8, hidden_dim=4, num_layers=2, dropout=0.0)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    z = m.encode(edge_index)
    assert tuple(z.shape) == (6, 4)
    scores = m.decode(z, torch.tensor([[0, 2], [3, 5]]))
    assert tuple(scores.shape) == (2,)


def test_split_partitions_without_overlap():
    edge_index = torch.arange(20).view(2, 10)
    tr, va, te = _split(edge_index, 0.2, 0.2, torch.Generator().manual_seed(0))
    assert tr.size(1) + va.size(1) + te.size(1) == 10
    assert va.size(1) == 2 and te.size(1) == 2


def test_negatives_are_typed_nonedges():
    group_ids = torch.tensor([0, 1])
    tech_ids = torch.tensor([2, 3, 4])
    existing = {(0, 2), (1, 3)}
    neg = _sample_negatives(5, group_ids, tech_ids, existing,
                            torch.Generator().manual_seed(0))
    assert neg.size(1) == 5
    for s, d in neg.t().tolist():
        assert s in (0, 1) and d in (2, 3, 4)   # group -> technique only
        assert (s, d) not in existing            # never a real edge


def test_undirected_doubles_edges():
    assert _undirected(torch.tensor([[0, 1], [1, 2]])).size(1) == 4
