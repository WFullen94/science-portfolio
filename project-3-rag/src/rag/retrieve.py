"""Stage 3a — Retrieval: two-stage bi-encoder recall + cross-encoder rerank.

Stage 1 (bi-encoder / FAISS): fast, approximate — pull top_k candidates.
Stage 2 (cross-encoder): slow, accurate — jointly encode (query, passage) pairs
and re-score, keeping the top rerank_top_n. This is the standard retrieval
pattern and adds a second transformer architecture (cross-encoder) to the stack.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from rag.config import load_config, resolve
from rag.index import build_embeddings


class TwoStageRetriever:
    def __init__(self, cfg):
        self.cfg = cfg
        r = cfg["retrieval"]
        self.top_k = r["top_k"]
        self.rerank_top_n = r["rerank_top_n"]
        self.store = FAISS.load_local(
            str(resolve(cfg["paths"]["faiss_index"])),
            build_embeddings(cfg),
            allow_dangerous_deserialization=True,  # our own local index
        )
        self.reranker = CrossEncoder(r["reranker"])

    def retrieve(self, query: str) -> list[Document]:
        candidates = self.store.similarity_search(query, k=self.top_k)
        if not candidates:
            return []
        pairs = [(query, d.page_content) for d in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda t: t[1], reverse=True)
        return [doc for doc, _ in ranked[: self.rerank_top_n]]


@lru_cache(maxsize=1)
def get_retriever() -> TwoStageRetriever:
    return TwoStageRetriever(load_config())


def format_context(docs: list[Document]) -> str:
    """Render retrieved chunks into a cited context block for the prompt."""
    blocks = []
    for i, d in enumerate(docs, 1):
        tid = d.metadata.get("technique_id", "?")
        name = d.metadata.get("name", "")
        blocks.append(f"[{i}] ({tid} — {name})\n{d.page_content}")
    return "\n\n".join(blocks)
