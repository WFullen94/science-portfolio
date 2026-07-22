# Project 3 — RAG + Evaluation over MITRE ATT&CK

A retrieval-augmented generation system over the **MITRE ATT&CK** threat-intelligence
knowledge base — built *and rigorously evaluated*. The evaluation is the point: most
candidates wire up a RAG chain; few measure faithfulness and retrieval quality.

```
ATT&CK STIX -> parse techniques -> chunk -> embed (bi-encoder) -> FAISS
   -> retrieve (top-k) -> rerank (cross-encoder) -> augment prompt -> generate
   -> evaluate with Ragas (faithfulness, context precision/recall)
   -> trace (Phoenix) -> serve behind an endpoint
```

## Stack (roadmap "lead with" tools)

| Stage | Tool |
|-------|------|
| Orchestration | **LangChain** |
| Embeddings | **sentence-transformers** (bi-encoder) |
| Reranking | **cross-encoder** (two-stage retrieval — a second transformer type) |
| Vector store | **FAISS** |
| Evaluation | **Ragas** — faithfulness, context precision/recall (reference-free) |
| Tracing | **Phoenix** (or Langfuse) |
| Serving | **FastAPI** endpoint (vLLM if self-hosting the generator) |

**Domain:** MITRE ATT&CK Enterprise — the canonical CTI corpus (~200+ techniques,
450+ sub-techniques). Also the corpus P4 (agent) and P6 (fine-tuning) build on.

## Layout

```
conf/config.yaml     paths, embedding/rerank models, chunking, retrieval, LLM backend
src/rag/
  config.py          config loader
  ingest.py          download + parse ATT&CK STIX -> technique documents   [commit 1]
  index.py           chunk + embed + build FAISS                           [commit 2]
  retrieve.py        two-stage retriever (bi-encoder + cross-encoder)      [commit 3]
  chain.py           RAG chain (retrieve -> augment -> generate)           [commit 3]
  evaluate.py        Ragas evaluation harness                             [commit 4]
  serve.py           FastAPI endpoint                                      [commit 5]
```

## Quickstart

```bash
cd project-3-rag
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# Local LLM backend (default): install Ollama, then
ollama pull llama3.1:8b

python -m rag.ingest      # download + parse ATT&CK  -> 697 technique docs
python -m rag.index       # chunk + embed + FAISS    -> 1,820 vectors
python -m rag.chain "How do adversaries dump credentials from LSASS?"   # ask it
python -m rag.evaluate    # Ragas: faithfulness + context precision/recall
uvicorn rag.serve:app --port 8000                                       # serve
RAG_TRACING=1 uvicorn rag.serve:app --port 8000                         # + Phoenix traces
```

## Evaluation (the part that matters)

`rag.evaluate` runs the chain over a curated question set
([eval/attack_questions.jsonl](eval/attack_questions.jsonl)) and scores it with **Ragas**:

| Metric | Measures | Diagnoses |
|--------|----------|-----------|
| Faithfulness | is the answer grounded in retrieved context? | **generator** (hallucination) |
| Response relevancy | does the answer address the question? | **generator** |
| Context precision | are relevant chunks retrieved + ranked high? | **retriever** |
| Context recall | did retrieval fetch everything the reference needs? | **retriever** |

Splitting retriever vs generator metrics tells you *which component to fix* — not just a score.
The LLM judge is the same pluggable backend as generation.

## Serving + tracing

`rag.serve` exposes `POST /query` (grounded answer + cited technique IDs) and `/health`, `/docs`.
Setting `RAG_TRACING=1` turns on **Phoenix** tracing — every retrieval and LLM call becomes an
OpenTelemetry span, so any single answer can be inspected end to end (which chunks, which prompt).
