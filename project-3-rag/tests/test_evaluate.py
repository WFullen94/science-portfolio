"""Eval-set loading test (no LLM/Ragas run)."""

from rag.config import load_config
from rag.evaluate import load_eval


def test_eval_set_well_formed():
    rows = load_eval(load_config())
    assert len(rows) >= 8
    for r in rows:
        assert r["user_input"].strip()
        assert r["reference"].strip()
        # Every reference should cite an ATT&CK technique id.
        assert "T1" in r["reference"]
