"""Drift-check logic tests (no Spark/Evidently run needed)."""

import json

from spine import drift
from spine.config import load_config, resolve


def test_extract_drift_parses_evidently_shape():
    fake = {"metrics": [
        {"result": {"some_other": 1}},
        {"result": {
            "share_of_drifted_columns": 0.42,
            "number_of_drifted_columns": 21,
            "number_of_columns": 49,
        }},
    ]}
    share, n_drift, n_cols = drift._extract_drift(fake)
    assert share == 0.42
    assert n_drift == 21
    assert n_cols == 49


def test_decide_branch_reads_result(tmp_path, monkeypatch):
    result_path = resolve(load_config()["paths"]["reports"]) / "drift_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    backup = result_path.read_text() if result_path.exists() else None
    try:
        result_path.write_text(json.dumps({"decision": drift.RETRAIN_TASK}))
        assert drift.decide_branch() == drift.RETRAIN_TASK
        result_path.write_text(json.dumps({"decision": drift.SKIP_TASK}))
        assert drift.decide_branch() == drift.SKIP_TASK
    finally:
        if backup is not None:
            result_path.write_text(backup)
        elif result_path.exists():
            result_path.unlink()
