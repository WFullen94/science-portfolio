"""RAG chain tests that don't require a running LLM."""

import pytest

from rag.chain import build_llm


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        build_llm({"generation": {
            "backend": "not-a-backend", "model": "x",
            "temperature": 0.0, "max_tokens": 8,
        }})


def test_ollama_backend_constructs():
    llm = build_llm({"generation": {
        "backend": "ollama", "model": "llama3.1:8b",
        "temperature": 0.0, "max_tokens": 8,
    }})
    # Construction only — no server call.
    assert llm.model == "llama3.1:8b"
