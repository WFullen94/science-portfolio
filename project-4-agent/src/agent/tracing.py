"""Phoenix tracing for the agent — reuses P3's shared setup.

The LangGraph agent runs LangChain runnables, so OpenInference's LangChain
auto-instrumentation captures the whole trajectory: each planning LLM call and
each tool execution becomes a span. That is how you inspect *why* an agent run
went the way it did — which tool it chose and what it saw back.

    AGENT_TRACING=1 python -m agent.graph "..."   # with a local Phoenix running
"""

from __future__ import annotations


def start_tracing(project_name: str = "attack-agent"):
    from rag.tracing import start_tracing as _shared  # same env as P3
    return _shared(project_name)
