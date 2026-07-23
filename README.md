# Science Portfolio — Applied Scientist (Databricks Public Sector)

[![P1 Spine CI](https://github.com/WFullen94/science-portfolio/actions/workflows/p1-ci.yml/badge.svg)](https://github.com/WFullen94/science-portfolio/actions/workflows/p1-ci.yml)
[![P3 RAG CI](https://github.com/WFullen94/science-portfolio/actions/workflows/p3-ci.yml/badge.svg)](https://github.com/WFullen94/science-portfolio/actions/workflows/p3-ci.yml)
[![Graph ML CI](https://github.com/WFullen94/science-portfolio/actions/workflows/graphml-ci.yml/badge.svg)](https://github.com/WFullen94/science-portfolio/actions/workflows/graphml-ci.yml)

Defensible, end-to-end artifacts built against the plan in
[Applied Scientist Roadmap.md](Applied%20Scientist%20Roadmap.md).

**Domain through-line:** Network Threat Detection & Threat Intelligence — everything here is
defensive/analytic (intrusion *detection*, telemetry analysis, CTI document work). No offensive tooling.

## Projects

| # | Name | Status | Resume claim |
|---|------|--------|--------------|
| 1 | [The Spine](project-1-spine/) — end-to-end ML on a lakehouse pattern | ✅ complete | Versioned data → tracked experiments → registered model → real-time endpoint → orchestrated retraining with data-validation gates and drift triggers |
| 3 | [RAG + Ragas over MITRE ATT&CK](project-3-rag/) | ✅ complete | Built *and rigorously evaluated* a RAG system over CTI — two-stage retrieval (bi-encoder + cross-encoder rerank), grounded generation, Ragas metrics (faithfulness + context precision/recall) that separate retriever from generator quality, served + traced |
| 4 | [Multi-tool threat-investigation agent + agent eval](project-4-agent/) | ✅ complete | Built *and evaluated* a multi-tool LangGraph agent (ATT&CK retriever + live NVD CVE lookup + technique mapping) — measured tool-selection accuracy (1.00) and trajectory validity (1.00), not just final output, with full Phoenix execution tracing |
| + | [Graph ML — link prediction on ATT&CK](project-graphml/) | ✅ complete | Trained a GraphSAGE GNN (PyTorch Geometric) on the ATT&CK knowledge graph to predict which techniques a threat group uses — ROC-AUC 0.908 on held-out edges with leakage-free splits and typed negative sampling |
| 5 | Probabilistic / conformal uncertainty on detection | ⬜ planned | — |
| 6 | LoRA/QLoRA fine-tune: CTI → ATT&CK (conditional) | ⬜ planned | — |
| 7 | Overhead object detection + distributed training | ⬜ planned | — |

Build order follows the roadmap: **P1 → P3 → P4 → P5 → (P6) → P7**. Ship each before opening the next.

## Continuous integration

Path-filtered GitHub Actions workflows ([.github/workflows/](.github/workflows/)) run each project's
test suite on push/PR — P1's Great Expectations validation gate + PySpark transforms, P3's
model-free tests (STIX parsing, chain wiring, eval-set integrity). They activate once the repo is
pushed to GitHub; the underlying test commands are verified locally.

## Philosophy

OSS-first, runs locally. Prove you understand what the managed platform does for you by building the
hand-rolled version (Spark/Delta/MLflow/FastAPI/Airflow), then note the managed equivalent
(Databricks Model Serving + Workflows). *A finished project beats ten named tools.*
