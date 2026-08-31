# Science Portfolio — Applied Scientist (Databricks Public Sector)

[![P1 Spine CI](https://github.com/WFullen94/science-portfolio/actions/workflows/p1-ci.yml/badge.svg)](https://github.com/WFullen94/science-portfolio/actions/workflows/p1-ci.yml)
[![P3 RAG CI](https://github.com/WFullen94/science-portfolio/actions/workflows/p3-ci.yml/badge.svg)](https://github.com/WFullen94/science-portfolio/actions/workflows/p3-ci.yml)
[![Graph ML CI](https://github.com/WFullen94/science-portfolio/actions/workflows/graphml-ci.yml/badge.svg)](https://github.com/WFullen94/science-portfolio/actions/workflows/graphml-ci.yml)
[![Causal CI](https://github.com/WFullen94/science-portfolio/actions/workflows/causal-ci.yml/badge.svg)](https://github.com/WFullen94/science-portfolio/actions/workflows/causal-ci.yml)
[![LoRA CI](https://github.com/WFullen94/science-portfolio/actions/workflows/lora-ci.yml/badge.svg)](https://github.com/WFullen94/science-portfolio/actions/workflows/lora-ci.yml)
[![Structured-DL CI](https://github.com/WFullen94/science-portfolio/actions/workflows/structdl-ci.yml/badge.svg)](https://github.com/WFullen94/science-portfolio/actions/workflows/structdl-ci.yml)
[![CV CI](https://github.com/WFullen94/science-portfolio/actions/workflows/cv-ci.yml/badge.svg)](https://github.com/WFullen94/science-portfolio/actions/workflows/cv-ci.yml)

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
| + | [Causal inference — does MFA reduce compromise?](project-causal/) | ✅ complete | Estimated a treatment effect from confounded observational data where the naive estimate flips sign — recovered the true ATE four ways (regression/IPW/AIPW/DoWhy) with placebo + random-common-cause refutations, and a T-learner for heterogeneous effects |
| 5 | Probabilistic / conformal uncertainty on detection | ⬜ planned | — |
| 6 | [LoRA fine-tune: CTI → ATT&CK](project-6-lora/) | ✅ complete | LoRA-fine-tuned an encoder to classify CTI procedure text into ATT&CK techniques — a controlled before/after (frozen linear probe 60% → LoRA 92% accuracy) training just 1.1% of parameters, tracked in MLflow |
| 7 | [Overwatch — overhead detection + SAHI tiling](project-7-cv/) | ✅ complete | Built an end-to-end overhead-detection service: fine-tuned YOLOv8 on VisDrone (aerial, tiny objects), then a *controlled* untiled-vs-SAHI-tiled head-to-head scored by one mAP implementation — SAHI lifts mAP@50 +0.022 and small-object recall +0.055 at 6.5× latency, quantifying the recall/latency trade rather than assuming it. ONNX export + FastAPI `/predict` (tiles at inference) + Docker + CI |
| + | [detsynth — incident → grounded ATT&CK detection pack](project-detection-synth/) | ✅ complete | Turns a MITRE ATT&CK technique into a *detection pack* (procedures, telemetry, Sigma starter, test fixtures) grounded in ATT&CK v17's real detection analytics. Showed model size is the dominant faithfulness lever (7B fixes wrong-Event-ID / wrong-OS errors a 1.5B makes) and ran a DPO faithfulness experiment with an independent eval — the gain didn't transfer, the honest result over a rigged win. **Capstone:** composed it with retrieval + P6's problem shape into incident → pack — maps free-text incidents to candidate techniques (measured mapping honestly at 23% top-1; let that number drive a human-confirmation-required serving design) and generates a pack conditioned on the incident's own specifics |
| 8 | [Structured-Data DL — transformers vs the right baseline](project-8-structured-dl/) | ✅ complete | Two *controlled* transformer-vs-baseline head-to-heads on NIDS data + distributed training. Tabular: a hand-rolled FT-Transformer (0.986) loses narrowly to tuned XGBoost (0.990) — trees still win on tables. Sequence: a time-series Transformer holds ~1.0 ROC-AUC while a GRU collapses to 0.61 at window length 256 — attention beats recurrence on long sequences. Distributed: DDP verified on CPU/gloo (2 ranks), FSDP wired + CUDA-gated. Lesson: match the inductive bias and *prove* it |
| + | [dfdetect — deepfake / synthetic-face detection](project-cv-deepfake/) | ✅ complete | CV learning survey, domain 1/8. Fine-tuned EfficientNet-B0 on real data spanning 40 deepfake generation techniques (DF40-derived); held-out test ROC-AUC 0.9963 / F1 0.9772, with the split held out as a genuinely separate download, not a random split of one file. Went looking for failures instead of stopping at the headline number — found one confidently-wrong real image that's the only JPEG in an otherwise-PNG pool, a plausible compression-artifact confound. ONNX-exported, served torch-free via ONNX Runtime, Dockerized, CI |

Build order follows the roadmap: **P1 → P3 → P4 → P5 → (P6) → P7 → P8**. Ship each before opening the next.

## Continuous integration

Path-filtered GitHub Actions workflows ([.github/workflows/](.github/workflows/)) run each project's
test suite on push/PR — P1's Great Expectations validation gate + PySpark transforms, P3's
model-free tests (STIX parsing, chain wiring, eval-set integrity). They activate once the repo is
pushed to GitHub; the underlying test commands are verified locally.

## Philosophy

OSS-first, runs locally. Prove you understand what the managed platform does for you by building the
hand-rolled version (Spark/Delta/MLflow/FastAPI/Airflow), then note the managed equivalent
(Databricks Model Serving + Workflows). *A finished project beats ten named tools.*
