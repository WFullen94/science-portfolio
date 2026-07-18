"""Serving-endpoint tests. Skips cleanly if no model has been trained yet."""

import warnings

import pytest

warnings.filterwarnings("ignore")


def _model_available() -> bool:
    import mlflow
    from mlflow.tracking import MlflowClient

    from spine.config import load_config

    cfg = load_config()
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    try:
        client = MlflowClient()
        return bool(client.search_model_versions(
            f"name='{cfg['mlflow']['registered_model']}'"))
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _model_available(),
    reason="no registered model yet — run `make train` first",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from spine.serve import app

    with TestClient(app) as c:
        yield c


def test_health_reports_loaded_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_name"]
    assert body["model_alias"] in ("staging", "candidate", "latest", "baked")


def test_predict_shapes_and_bounds(client):
    flows = [
        {"IN_BYTES": 120, "IN_PKTS": 60, "OUT_BYTES": 40, "OUT_PKTS": 2,
         "PROTOCOL": 6, "TCP_FLAGS": 40, "FLOW_DURATION_MILLISECONDS": 3},
        {"IN_BYTES": 2000, "IN_PKTS": 8, "OUT_BYTES": 4000, "OUT_PKTS": 9},
    ]
    body = client.post("/predict", json={"flows": flows}).json()
    assert len(body["predictions"]) == 2
    for p in body["predictions"]:
        assert 0.0 <= p["malicious_probability"] <= 1.0
        assert p["prediction"] in (0, 1)
        assert p["verdict"] in ("attack", "benign")


def test_empty_flows_rejected(client):
    assert client.post("/predict", json={"flows": []}).status_code == 422
