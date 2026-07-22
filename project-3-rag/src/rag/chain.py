"""Stage 3b — RAG chain: retrieve -> augment prompt -> generate (grounded).

Ties the two-stage retriever to a pluggable LLM (Ollama by default; Anthropic /
OpenAI if configured). The prompt constrains the model to answer only from the
retrieved ATT&CK context and cite technique IDs — the grounding that Ragas then
measures (faithfulness / context precision-recall).
"""

from __future__ import annotations

import sys
from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from opentelemetry import trace

from rag.config import load_config
from rag.retrieve import format_context, get_retriever

# No-op tracer unless a provider is registered (rag.tracing.start_tracing);
# when tracing is on, this yields a rag.query -> rag.retrieve + LLM span tree.
_tracer = trace.get_tracer("rag")

SYSTEM = (
    "You are a cyber threat intelligence analyst. Answer the question using ONLY "
    "the MITRE ATT&CK context provided below. Cite the relevant technique IDs "
    "(e.g. T1566) inline. If the context does not contain the answer, say you do "
    "not have enough information — never use outside knowledge."
)


def build_llm(cfg):
    """Pluggable chat model. Only the selected backend's package must be installed."""
    g = cfg["generation"]
    backend = g["backend"]
    if backend == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=g["model"], temperature=g["temperature"],
                          num_predict=g["max_tokens"])
    if backend == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=g["model"], temperature=g["temperature"],
                             max_tokens=g["max_tokens"])
    if backend == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=g["model"], temperature=g["temperature"],
                          max_tokens=g["max_tokens"])
    raise ValueError(f"unknown generation backend: {backend!r}")


class RagChain:
    def __init__(self, cfg=None):
        self.cfg = cfg or load_config()
        self.retriever = get_retriever()
        self.llm = build_llm(self.cfg)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ])

    def invoke(self, question: str) -> dict:
        with _tracer.start_as_current_span("rag.query") as span:
            span.set_attribute("rag.question", question)
            with _tracer.start_as_current_span("rag.retrieve") as rspan:
                docs = self.retriever.retrieve(question)
                rspan.set_attribute("rag.n_retrieved", len(docs))
            messages = self.prompt.format_messages(
                context=format_context(docs), question=question
            )
            response = self.llm.invoke(messages)  # LLM span nests here when traced
            techniques = [d.metadata.get("technique_id") for d in docs]
            span.set_attribute("rag.source_techniques", ",".join(t for t in techniques if t))
            return {
                "question": question,
                "answer": response.content,
                "contexts": [d.page_content for d in docs],
                "source_techniques": techniques,
            }


@lru_cache(maxsize=1)
def get_chain() -> RagChain:
    return RagChain()


def main() -> int:
    q = " ".join(sys.argv[1:]) or "How do adversaries abuse valid accounts to persist?"
    out = get_chain().invoke(q)
    print(f"Q: {out['question']}\n")
    print(f"ANSWER:\n{out['answer']}\n")
    print(f"SOURCES: {out['source_techniques']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
