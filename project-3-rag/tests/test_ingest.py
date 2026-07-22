"""ATT&CK parsing tests — no network needed (uses a tiny STIX fixture)."""

from rag.ingest import parse_techniques

_FIXTURE = {"objects": [
    {  # active technique
        "type": "attack-pattern",
        "name": "Phishing",
        "description": "Adversaries may send phishing messages.",
        "kill_chain_phases": [{"kill_chain_name": "mitre-attack", "phase_name": "initial-access"}],
        "x_mitre_platforms": ["Windows", "macOS"],
        "x_mitre_is_subtechnique": False,
        "external_references": [
            {"source_name": "mitre-attack", "external_id": "T1566", "url": "https://attack.mitre.org/techniques/T1566"},
        ],
    },
    {  # deprecated -> excluded
        "type": "attack-pattern", "name": "Old", "description": "x",
        "x_mitre_deprecated": True,
        "external_references": [{"source_name": "mitre-attack", "external_id": "T9999"}],
    },
    {"type": "relationship"},  # non-technique -> ignored
]}


def test_parses_only_active_techniques():
    docs = parse_techniques(_FIXTURE)
    assert len(docs) == 1
    d = docs[0]
    assert d["technique_id"] == "T1566"
    assert d["tactics"] == ["initial-access"]
    assert "Windows" in d["platforms"]


def test_document_text_has_id_name_and_description():
    d = parse_techniques(_FIXTURE)[0]
    assert "T1566" in d["text"]
    assert "Phishing" in d["text"]
    assert "phishing messages" in d["text"]
