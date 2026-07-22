"""Stage 5 — Serve: the RAG chain behind a FastAPI endpoint.

POST /query runs retrieve -> rerank -> generate and returns a grounded answer
with the ATT&CK techniques it drew from. Optional Phoenix tracing (enabled via
RAG_TRACING=1) records each retrieval + LLM call for observability.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag.chain import get_chain


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.environ.get("RAG_TRACING") == "1":
        from rag.tracing import start_tracing
        start_tracing()
    get_chain()  # warm the retriever + LLM at startup
    yield


app = FastAPI(
    title="ATT&CK RAG",
    description="Retrieval-augmented Q&A over the MITRE ATT&CK knowledge base.",
    version="1.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    question: str = Field(..., description="A question about adversary TTPs.")
    include_contexts: bool = Field(False, description="Return the retrieved passages too.")

    model_config = {"json_schema_extra": {"example": {
        "question": "How do adversaries steal credentials from LSASS memory?",
    }}}


class QueryResponse(BaseModel):
    question: str
    answer: str
    source_techniques: list[str]
    contexts: list[str] | None = None


@app.get("/")
def root():
    return {"service": "attack-rag", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(422, "empty question")
    out = get_chain().invoke(req.question)
    return QueryResponse(
        question=out["question"],
        answer=out["answer"],
        source_techniques=out["source_techniques"],
        contexts=out["contexts"] if req.include_contexts else None,
    )
