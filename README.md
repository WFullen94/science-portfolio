# Science Portfolio — Applied Scientist (Databricks Public Sector)

Defensible, end-to-end artifacts built against the plan in
[Applied Scientist Roadmap.md](Applied%20Scientist%20Roadmap.md).

**Domain through-line:** Network Threat Detection & Threat Intelligence — everything here is
defensive/analytic (intrusion *detection*, telemetry analysis, CTI document work). No offensive tooling.

## Projects

| # | Name | Status | Resume claim |
|---|------|--------|--------------|
| 1 | [The Spine](project-1-spine/) — end-to-end ML on a lakehouse pattern | ✅ complete | Versioned data → tracked experiments → registered model → real-time endpoint → orchestrated retraining with data-validation gates and drift triggers |
| 3 | [RAG + Ragas over MITRE ATT&CK](project-3-rag/) | ✅ complete | Built *and rigorously evaluated* a RAG system over CTI — two-stage retrieval (bi-encoder + cross-encoder rerank), grounded generation, Ragas metrics (faithfulness + context precision/recall) that separate retriever from generator quality, served + traced |
| 4 | Multi-tool threat-investigation agent + agent eval | ⬜ planned | — |
| 5 | Probabilistic / conformal uncertainty on detection | ⬜ planned | — |
| 6 | LoRA/QLoRA fine-tune: CTI → ATT&CK (conditional) | ⬜ planned | — |
| 7 | Overhead object detection + distributed training | ⬜ planned | — |

Build order follows the roadmap: **P1 → P3 → P4 → P5 → (P6) → P7**. Ship each before opening the next.

## Philosophy

OSS-first, runs locally. Prove you understand what the managed platform does for you by building the
hand-rolled version (Spark/Delta/MLflow/FastAPI/Airflow), then note the managed equivalent
(Databricks Model Serving + Workflows). *A finished project beats ten named tools.*
