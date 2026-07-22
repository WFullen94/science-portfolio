"""Phoenix tracing — per-request observability for the RAG chain.

Registers OpenInference auto-instrumentation for LangChain so every retrieval and
LLM call becomes an OpenTelemetry span, viewable in the Phoenix UI. Traces are
how you debug RAG in production: a bad answer is almost always a bad retrieval,
and the trace shows exactly which chunks were fetched.

    import phoenix as px; px.launch_app()   # local UI at http://localhost:6006
    RAG_TRACING=1 uvicorn rag.serve:app      # serve with tracing on
"""

from __future__ import annotations

import os


def start_tracing(project_name: str = "attack-rag"):
    """Wire LangChain -> OpenTelemetry -> Phoenix. No-op-safe if deps missing."""
    try:
        from phoenix.otel import register
    except ImportError:
        print("[tracing] phoenix not installed — skipping (pip install arize-phoenix)")
        return None

    endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")
    tracer_provider = register(
        project_name=project_name,
        endpoint=f"{endpoint}/v1/traces",
        auto_instrument=True,  # picks up openinference-instrumentation-langchain
    )
    print(f"[tracing] Phoenix tracing on -> {endpoint} (project={project_name})")
    return tracer_provider
