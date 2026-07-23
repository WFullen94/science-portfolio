"""Prediction-helper tests on a tiny fake graph (no trained model needed)."""

import torch

from attackgraph.build_graph import GROUP, TECHNIQUE
from attackgraph.predict import find_group, rank_techniques


def _fake_graph():
    return {
        "node_type": torch.tensor([GROUP, TECHNIQUE, TECHNIQUE]),
        "names": ["APT-Test", "Tech One", "Tech Two"],
        "ext_ids": ["", "T1001", "T1002"],
        "target_edges": torch.tensor([[0], [1]]),  # group 0 uses technique node 1
    }


def test_find_group_case_insensitive():
    matches = find_group(_fake_graph(), "apt-test")
    assert matches and matches[0][0] == 0


def test_rank_marks_known_edges():
    g = _fake_graph()
    ranked = rank_techniques(g, torch.eye(3), group_idx=0)
    assert len(ranked) == 2
    by_id = {r["technique_id"]: r for r in ranked}
    assert by_id["T1001"]["known"] is True    # documented edge
    assert by_id["T1002"]["known"] is False   # candidate prediction
