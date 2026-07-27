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
- 🚧 **Structured-Data DL** — *next*: tabular + time-series transformers, distributed (DDP→FSDP).
- 🟡 **DPO / RLHF (extends P6)** — *added, gated*: preference-tune the P6 model. Feasible here
  (LoRA + DPO, slow on MPS); DPO is far lighter than PPO-RLHF.
- 🟡 **Autonomous cyber-defense RL (CAGE / CybORG)** — *added, gated*: an RL agent defends a
  simulated network. On-domain (threat defense as sequential decision-making); CPU-trainable here.

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

*Order is a default, not a contract — clusters can be resequenced. The ATT&CK corpus (P3) feeds
P4, Graph ML, and P6; the deep-learning items (8, 9) are the finale; the RL items (11, 12) are
gated additions — build after the core plan ships. Both are buildable on this machine (CAGE on CPU,
DPO on MPS); neither needs a GPU box (unlike distributed training / QLoRA).*

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

## The one rule (from the roadmap)

**Ship before adding.** The plan was reopened once, deliberately, to add two RL projects (11, 12) —
now re-closed. The binding constraint is no longer coverage — it's finishing. RL items are **gated**:
build them only after the core plan ships. A finished project beats a longer plan. Backlog for
anything further: name-drop only (recommenders, contextual bandits, ONNX/Triton, K8s).
