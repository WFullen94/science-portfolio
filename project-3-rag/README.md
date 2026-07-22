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
python -m rag.ingest      # download + parse ATT&CK
python -m rag.index       # build the FAISS index
```
