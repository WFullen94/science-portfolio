"""Graph-construction tests on a tiny STIX fixture (no network)."""

from attackgraph.build_graph import build_graph

FIX = {"objects": [
    {"id": "attack-pattern--t1", "type": "attack-pattern", "name": "T-A",
     "external_references": [{"source_name": "mitre-attack", "external_id": "T1001"}]},
    {"id": "attack-pattern--t2", "type": "attack-pattern", "name": "T-B",
     "external_references": [{"source_name": "mitre-attack", "external_id": "T1001.001"}]},
    {"id": "intrusion-set--g1", "type": "intrusion-set", "name": "APT-X"},
    {"id": "malware--m1", "type": "malware", "name": "BadWare"},
    {"id": "attack-pattern--dep", "type": "attack-pattern", "name": "Old",
     "x_mitre_deprecated": True, "external_references": []},
    {"id": "rel--1", "type": "relationship", "relationship_type": "uses",
     "source_ref": "intrusion-set--g1", "target_ref": "attack-pattern--t1"},
    {"id": "rel--2", "type": "relationship", "relationship_type": "uses",
     "source_ref": "malware--m1", "target_ref": "attack-pattern--t2"},
    {"id": "rel--3", "type": "relationship", "relationship_type": "uses",
     "source_ref": "intrusion-set--g1", "target_ref": "malware--m1"},
    {"id": "rel--4", "type": "relationship", "relationship_type": "subtechnique-of",
     "source_ref": "attack-pattern--t2", "target_ref": "attack-pattern--t1"},
]}


def test_node_typing_and_active_filter():
    g = build_graph(FIX)
    assert g["num_nodes"] == 4  # deprecated technique excluded
    assert g["counts"]["technique"] == 2
    assert g["counts"]["group"] == 1
    assert g["counts"]["software"] == 1


def test_edge_routing():
    g = build_graph(FIX)
    # group->technique is the prediction target; the rest are auxiliary.
    assert g["counts"]["group_uses_technique"] == 1
    assert tuple(g["target_edges"].shape) == (2, 1)
    assert g["counts"]["software_uses_technique"] == 1
    assert g["counts"]["group_uses_software"] == 1
    assert g["counts"]["subtechnique_of"] == 1
    assert g["aux_edges"].shape[1] == 3
