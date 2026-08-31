# Portfolio Master Plan

The living sequence. Domain through-line: **network threat detection & CTI**. Built OSS-first,
one project shipped before the next is opened. Derived from
[Applied Scientist Roadmap.md](Applied%20Scientist%20Roadmap.md) + agreed additions.

## Status

- ✅ **P1 — The Spine** — end-to-end NIDS ML lifecycle (Delta → PySpark → XGBoost/Optuna/MLflow
  → SHAP → FastAPI/Docker → Airflow + Evidently drift). *Shipped.*
- ✅ **P3 — RAG over MITRE ATT&CK** — ingest → FAISS → two-stage retrieval (bi-encoder +
  cross-encoder) → grounded generation (local Ollama) → Ragas eval → FastAPI + Phoenix tracing.
  *Shipped.* Ragas: context precision 1.00 / recall 0.98 (retriever), faithfulness 0.77.
- ✅ **CI/CD (GitHub Actions)** — path-filtered workflows run each project's tests on push. *Shipped.*
- ✅ **P4 — Threat-investigation agent** — 3 tools (ATT&CK retriever + live NVD CVE lookup +
  technique mapping) → LangGraph ReAct agent → agent eval (tool-selection 1.0 / trajectory 1.0 /
  completion 0.875) → Phoenix tracing. *Shipped.*
- ✅ **Graph ML / GNN on ATT&CK** — GraphSAGE link prediction (PyG): predicts group→technique
  edges, ROC-AUC 0.908 held-out, with a threat-group TTP-prediction demo. *Shipped.*
- ⏸️ **P5 — Uncertainty (Bayesian + conformal)** — deferred by request; circle back later.
- ✅ **Causal inference** — MFA→compromise: naive estimate flips sign under confounding; recovered
  the true ATE (regression/IPW/AIPW/DoWhy) + refutations + T-learner CATE. *Shipped.*
- ✅ **P6 — LoRA fine-tuning** — CTI→ATT&CK technique classification; linear-probe 60% vs LoRA 92%
  accuracy training 1.1% of params (DistilBERT + PEFT), MLflow-tracked. *Shipped.*
- ✅ **P8 — Structured-Data DL (complete)** — two controlled transformer-vs-baseline head-to-heads
  plus distributed training. *Stage 1 (tabular):* XGBoost 0.990 edges the FT-Transformer's 0.986 —
  trees still win on tabular. *Stage 2 (sequence):* a time-series Transformer holds ~1.0 ROC-AUC
  while a GRU collapses to 0.607 (near-random) at window length 256 — attention beats recurrence on
  long sequences. *Stage 3 (distributed):* DDP verified end-to-end on CPU/gloo (2 ranks, 0.9985);
  FSDP wired (shard + FULL_STATE_DICT) and CUDA-gated. Same lesson throughout: match the inductive
  bias, prove it. *Shipped.*
- ✅ **P7 — Overwatch (overhead CV detection)** — fine-tuned YOLOv8n on VisDrone (aerial, tiny
  objects), then a controlled untiled-vs-SAHI-tiled head-to-head scored by one mAP implementation:
  SAHI lifts mAP@50 +0.022 and small-object recall +0.055 at 6.5× latency. ONNX export + FastAPI
  `/predict` (tiles at inference) + Dockerfile + CI. *Shipped.*
- ✅ **detsynth — grounded ATT&CK detection-content generator** (item 11, pivoted from "DPO on P6").
  Turns a technique into a *detection pack* (procedures, telemetry, Sigma starter, test fixtures)
  grounded in ATT&CK v17's real detection analytics. *Phase 1:* STIX grounding + generator; 7B fixes
  the hallucinations a 1.5B makes. *Phase 2:* unblocked (pinned transformers 4.46/TRL 0.12/peft 0.13
  — the bleeding-edge stack hung on MPS) and run — DPO improved the metric it optimizes (grounding
  margin 0.079→0.089) but that gain didn't transfer to an independent platform-faithfulness check
  (1.00→0.97), the honest circularity result. *Capstone:* composed it with retrieval + P6's problem
  shape into "incident → pack" — map an incident to candidate techniques (retrieval over all 697
  grounded techniques, not just P6's 20), then generate a pack conditioned on the incident's
  specifics. Measured mapping honestly (23% top-1; two standard fixes — analytic-text enrichment,
  cross-encoder rerank — didn't help) and let that number drive the serving design: human
  confirmation required by default, auto-generate only on explicit opt-in and always labeled
  provisional. *Shipped, all phases complete.*
  (map incident→technique via P6/retrieval, condition generation on the incident, reuse this DPO).
- 🟡 **Autonomous cyber-defense RL (CAGE / CybORG)** — *added, gated*: an RL agent defends a
  simulated network. On-domain (threat defense as sequential decision-making); CPU-trainable here.

### Added candidates (gated — from the Aug-2026 workspace survey)

A sweep of the wider `~` workspace found deep *notebook* coverage of CV, multimodal, serving, and
forecasting but **no end-to-end systems** for them (only `science-portfolio` + `llm-reliability` +
`agent-eval` are real systems). These promote that knowledge into shippable services. All **gated**:
build after P8 stages 2–3 and the standing core. New *system archetypes* the portfolio lacks are
called out — breadth here is about engineering shape, not just another model.

- 🔵 **Forecasting system** — dedicated time-series forecasting: classical (ARIMA/ETS/Prophet) vs
  deep (PatchTST / N-BEATS / TFT), **rolling-origin backtesting**, probabilistic/quantile output,
  MLflow + serving. On-domain option: forecast network-traffic / attack volume. *(Deeper than P8's
  single time-series-transformer stage; `time-series-forecasting` is the deepest notebook set.)*
- 🔵 **Serving / deployment flagship** — one model served **three ways** (FastAPI vs vLLM/Triton vs
  ONNX/TensorRT) with a shared load-test harness and a p50/p95 latency-throughput-cost report. Fills
  the entirely-absent GPU/optimized-serving category; reuses the P1 model. *New archetype.*
- 🔵 **Multimodal CLIP image-search service** — promote `representation-learning/15_cross_modal_
  retrieval` into a real service: `open_clip` encoder + persistent FAISS/qdrant index + FastAPI
  `/search` + build-index CLI + recall@k test + Dockerfile. Lowest-lift multimodal system.
- 🔵 **Recommender (two-tower retrieval + ranking)** — *new archetype*: candidate generation +
  reranking, offline↔online parity. Most industry-ubiquitous shape the portfolio lacks. On-domain:
  "techniques/threats similar to this one."
- 🔵 **Active-learning loop** — *new archetype*: uncertainty sampling → label → retrain feedback
  cycle (human-in-the-loop). Sits on the deep `data-centric-ai` notebooks; pairs with P5 uncertainty.
  On-domain: prioritize which alerts an analyst labels next.
- ⚪ **Opportunistic / on-domain extensions** (lighter): standalone model-monitoring service,
  streaming anomaly detection, contextual bandits (adaptive triage), CTI→knowledge-graph construction
  from raw reports, feature store + train/serve parity.

## Sequence

| # | Project | Type | Key skills | Reuses |
|---|---------|------|-----------|--------|
| 1 | **P3 — RAG + Ragas** over ATT&CK | core | LangChain, FAISS, bi-encoder + **cross-encoder rerank**, Ragas eval, Phoenix tracing, serving | — |
| 2 | **CI/CD (GitHub Actions)** | fold-in | tests + data-validation on push; MLE table-stakes | P1 + P3 |
| 3 | **P4 — Threat-investigation agent** | core | LangGraph, tool calling, structured outputs, agent eval | P3 corpus |
| 4 | **Graph ML / GNN on ATT&CK** | added | GNN, knowledge graph (ATT&CK = 21k relationships), attack-path modeling | P3/P4 corpus |
| 5 | **P5 — Uncertainty** | core | PyMC (Bayesian/MCMC), conformal prediction (MAPIE) | P1 data |
| 6 | **Experimentation / causal inference** | added | experiment design, treatment-effect estimation, confounding | pairs w/ P5 |
| 7 | **P6 — Fine-tuning** | core | LoRA/QLoRA, **encoder-only vs decoder-only**, HF Transformers/Accelerate | P3 corpus |
| 8 | **Structured-Data DL + Distributed** | added | **tabular transformer** vs XGBoost, **time-series transformer**, PyTorch DDP→FSDP | fresh datasets |
| 9 | **P7 — Overhead CV detection** | core | YOLO/R-CNN, SAHI tiling, mAP, distributed training | — |
| 10 | **Streaming / real-time inference** | added | Spark Structured Streaming / Kafka real-time scoring | P1 extension |
| 11 | **DPO / RLHF — align the P6 model** | added (RL) | preference tuning (DPO), reward-model concepts, extends LoRA | P6 model + P3 corpus |
| 12 | **Autonomous cyber-defense RL (CAGE/CybORG)** | added (RL) | deep RL (PPO/DQN), sequential decision-making, gym env, reward design | on-domain (network defense) |
| 13 | **Forecasting system** | gated | classical vs deep (PatchTST/N-BEATS/TFT), backtesting, probabilistic | fresh / P1 traffic |
| 14 | **Serving / deployment flagship** | gated | vLLM/Triton/ONNX/TensorRT, load testing, latency-throughput report | P1 model |
| 15 | **Multimodal CLIP image-search service** | gated | open_clip, FAISS/qdrant, FastAPI, recall@k | promotes RL notebook |
| 16 | **Recommender (two-tower)** | gated | retrieval + ranking, offline↔online parity | — |
| 17 | **Active-learning loop** | gated | uncertainty sampling, human-in-the-loop retrain cycle | P1 data + P5 |

*Order is a default, not a contract — clusters can be resequenced. The ATT&CK corpus (P3) feeds
P4, Graph ML, and P6; the deep-learning items (8, 9) are the finale; the RL items (11, 12) are
gated additions — build after the core plan ships. Items 13–17 are gated candidates from the
Aug-2026 workspace survey (notebooks → systems); several are new archetypes the portfolio lacks
(serving, recsys, active-learning loop). Most run on this machine; the serving flagship's vLLM/Triton
and P8/P7 distributed training want a GPU box.*

## Skill coverage — what each competency maps to

| Competency | Where |
|-----------|-------|
| Classical ML lifecycle / MLOps | P1 |
| Distributed data (Spark/Delta) | P1 |
| Experiment tracking + registry (MLflow) | P1, P6, P8 |
| Serving (REST/Docker; vLLM for LLMs) | P1, P3 |
| Orchestration + drift retraining | P1 |
| RAG + retrieval eval | P3 |
| Agents + agent eval | P4 |
| Graph ML / GNNs | Graph project |
| Bayesian + conformal uncertainty | P5 |
| Experimentation + causal inference | Causal project |
| LLM fine-tuning (PEFT/LoRA) | P6 |
| Transformers — encoder / decoder / embedding / cross-encoder / vision | P3, P6, P7, P8 |
| Transformers over tabular + time-series | P8 |
| Distributed training (DDP→FSDP) | P8, P7 |
| CV (detection/segmentation/change) | P7 + CV track |
| Streaming / real-time ML | Streaming ext. |
| CI/CD for ML | CI/CD fold-in |
| Reinforcement learning (DPO/RLHF, deep RL) | DPO project + CAGE project |
| Forecasting (classical + deep, backtesting) | Forecasting system (13) |
| Optimized/GPU serving (vLLM/Triton/ONNX/TensorRT) | Serving flagship (14) |
| Multimodal / cross-modal retrieval (CLIP) | CLIP image-search (15) |
| Recommenders (two-tower retrieval + ranking) | Recommender (16) |
| Active learning / human-in-the-loop | Active-learning loop (17) |

## The one rule (from the roadmap)

**Ship before adding.** The plan was reopened to add the RL projects (11, 12) and again to log the
Aug-2026 survey candidates (13–17). The binding constraint is not coverage — it's finishing. Items
11–17 are all **gated**: build only after the core plan and P8 stages ship, one at a time. A finished
project beats a longer plan. Backlog beyond this: name-drop only (K8s/KServe, feature stores,
contextual bandits, CTI knowledge-graph construction, model-monitoring service).
