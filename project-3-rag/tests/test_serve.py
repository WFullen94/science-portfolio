"""Serving tests that don't require the LLM server. Skip if no FAISS index yet."""

import warnings

import pytest

from rag.config import load_config, resolve

warnings.filterwarnings("ignore")

pytestmark = pytest.mark.skipif(
    not (resolve(load_config()["paths"]["faiss_index"]) / "index.faiss").exists(),
    reason="no FAISS index yet — run `python -m rag.index` first",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from rag.serve import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_root_lists_endpoints(client):
    body = client.get("/").json()
    assert body["service"] == "attack-rag"


def test_empty_question_rejected(client):
    # Rejected before any LLM call, so this needs no model server.
    assert client.post("/query", json={"question": "   "}).status_code == 422
