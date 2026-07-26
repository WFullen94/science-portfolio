"""Dataset-construction tests (STIX fixture, no network)."""

from ctilora.build_dataset import _clean, build_examples


def test_clean_strips_markdown_links_and_citations():
    out = _clean("[APT29](https://x) used [Cobalt Strike](https://y) to inject. (Citation: Foo)")
    assert "http" not in out
    assert "Citation" not in out
    assert "APT29 used Cobalt Strike to inject." in out


FIX = {"objects": [
    {"id": "attack-pattern--a", "type": "attack-pattern", "name": "Process Injection",
     "external_references": [{"source_name": "mitre-attack", "external_id": "T1055"}]},
    {"id": "intrusion-set--g", "type": "intrusion-set", "name": "APT-X"},
    {"id": "rel--1", "type": "relationship", "relationship_type": "uses",
     "source_ref": "intrusion-set--g", "target_ref": "attack-pattern--a",
     "description": "[APT-X](u) injected code into a running process to evade defenses."},
    {"id": "rel--2", "type": "relationship", "relationship_type": "uses",
     "source_ref": "intrusion-set--g", "target_ref": "attack-pattern--a"},  # no desc -> skipped
]}


def test_build_examples_labels_text_with_technique():
    df = build_examples(FIX)
    assert len(df) == 1                        # the description-less relationship is dropped
    assert df.iloc[0]["technique_id"] == "T1055"
    assert "http" not in df.iloc[0]["text"]
