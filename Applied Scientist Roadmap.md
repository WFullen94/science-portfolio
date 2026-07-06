# Applied Scientist Roadmap — Databricks Public Sector

A consolidated build plan. The goal is defensible artifacts, not a long skills list.
Every tool below earns its place inside a project you can talk about under questioning.

---

## The one rule

You surfaced 50+ tools in exploration. That was the right way to map the space.
Now the binding constraint flips: it is no longer how many tools you can name,
it is how many you can defend in an interview and show in a project. Those are small numbers.

Stop adding. Start building. Everything past this plan is a distraction until the projects exist.

---

## Tool tiers (final cut)

### Lead with these — demonstrated in projects, safe to be drilled on

- **MLflow** — tracking, registry, model packaging
- **PySpark** — distributed data processing
- **Delta Lake** — versioned, ACID tables
- **PyTorch (DDP → FSDP)** — distributed training
- **RAG stack:** LangChain + FAISS/Chroma
- **Ragas** — RAG evaluation
- **Optuna** — hyperparameter search
- **SHAP** — explainability
- **XGBoost / scikit-learn** — classical ML (still wins most tabular problems)
- **Great Expectations (or Pandera)** — data validation
- **Model Serving** — Databricks Model Serving (or FastAPI + Docker); vLLM for LLM serving
- **Orchestration** — Databricks Workflows (or Airflow) — scheduled DAGs, retraining triggers
- **Agent orchestration** — LangGraph (or Mosaic AI Agent Framework) + agent evaluation

### List as familiar — concept + light hands-on, can hold a short conversation

- **Ray Tune** — distributed hyperparameter search
- **HF Accelerate** — single/multi-GPU abstraction
- **Unity Catalog** (concepts: catalog/schema/table, lineage, governance)
- **dbt** — SQL transformations
- **Evidently** — drift / model monitoring
- **PyMC** — probabilistic modeling
- **Langfuse or Phoenix** — LLM / agent observability
- **Weights & Biases** — alternative experiment tracker
- **DSPy** — programming (not prompting) LLMs
- **Conformal prediction (MAPIE)** — calibrated prediction intervals
- **Docker** — containerization (table stakes for serving)
- **Feature Store** — Databricks Feature Store (native); Feast (OSS)
- **Structured outputs / function calling** — Pydantic + constrained decoding
- **SQL** — the actual daily-driver skill; easy to under-weight
- **PEFT / LoRA / QLoRA** — parameter-efficient fine-tuning (Project 6, conditional)

### Know it exists — name-drop only if directly relevant

DeepEval, TruLens, MLflow LLM eval, NumPyro, BoTorch/Ax, DeepSpeed, Dask, Feast, DVC, Hydra,
pytest-for-ML, Dagster/Prefect (orchestration alternatives)

### Compliance literacy — reading, not coding, but a real public-sector differentiator

FedRAMP · IL4/IL5 · NIST 800-53 / FISMA · NIST AI Risk Management Framework ·
how Unity Catalog governance (lineage, access control, audit) maps to these

### Supporting evidence, not headline

Triton / TensorRT / GPU optimization — reframe as "I can optimize the per-node model,
then distribute it," not as your core pitch.

---

## The projects

Five artifacts. Together they demonstrate every "lead with" tool and most "familiar" ones,
each in a context you can defend. The full lifecycle:
**data → train → evaluate → serve → orchestrate → agent-ify → observe.**

### Project 1 — The Spine (core end-to-end ML on the platform pattern)

**Build:** Load a public dataset into a Delta table → feature-engineer in PySpark
→ train XGBoost with MLflow autolog → compare runs in the MLflow UI
→ register the best model and promote it → reload and score new data.
→ **serve** the registered model behind a real-time endpoint (Databricks Model Serving,
or FastAPI + Docker) → **orchestrate** the whole thing as a scheduled DAG with a
drift-triggered retraining path.

Add-ons that each demonstrate a category, no new project:

- Great Expectations validation suite on the input data
- SHAP explanation of the final model
- Optuna for the hyperparameter search step
- Evidently drift check wired to the retraining trigger

**Tools shown:** Delta, PySpark, MLflow (tracking + registry), XGBoost, Great Expectations,
SHAP, Optuna, Model Serving, Workflows/Airflow, Evidently, Docker

**Resume claim:** End-to-end ML pipeline on a lakehouse pattern — versioned data, tracked
experiments, registered model served behind a real-time endpoint, orchestrated as a scheduled
workflow with data-validation gates and a drift-triggered retraining path.

### Project 2 — Distributed Training

**Build:** Take one model (a neural net on a dataset that benefits from scale).
Train it with PyTorch DDP, then convert to FSDP (or HF Accelerate).
Stream data efficiently rather than loading it all. Track with MLflow.
Write up why you scaled and what changed at each rung.

**Tools shown:** PyTorch DDP, FSDP, Accelerate, MLflow, (streaming data concept)

**Resume claim:** Scaled model training from single-GPU to distributed data-parallel,
with a measured rationale for each step up the ladder.

### Project 3 — RAG Application + Evaluation

**Build:** Chunk → embed → store in FAISS/Chroma → retrieve → augment prompt,
orchestrated with LangChain. Then evaluate it with Ragas (faithfulness,
context precision/recall) — this is the part most candidates skip and the part that impresses.
Add Langfuse (or Phoenix) tracing for the observability angle.
Put the retrieval+generation behind an endpoint (vLLM if self-hosting the generator,
otherwise the app endpoint itself).

**Tools shown:** LangChain, FAISS/Chroma, Ragas, Langfuse/Phoenix, serving (vLLM/endpoint),
(DSPy optional)

**Resume claim:** Built and rigorously evaluated a RAG system — not just wired it up,
but measured faithfulness and retrieval quality with reference-free metrics, and served it
behind a real endpoint.

### Project 4 — Agent + Agent Evaluation

**Build:** Take the RAG retriever from Project 3 and wrap it as *one tool* among several.
Give an agent 2–3 tools (retriever + a calculator or structured-data lookup + maybe a web/API call),
let it plan and call them. Orchestrate with LangGraph or the Mosaic AI Agent Framework directly.
Then — the part that matters — **evaluate the agent**: not just final-answer quality, but
tool-selection correctness, trajectory/step validity, and task completion. Add tracing
(Langfuse/Phoenix, already in your stack) so you can inspect why a run went the way it did.

**Tools shown:** LangGraph (or Mosaic AI Agent Framework), tool/function calling,
structured outputs (Pydantic), agent evaluation (trajectory + tool-choice metrics),
Langfuse/Phoenix tracing

**Resume claim:** Built and evaluated a multi-tool agent — measured tool-selection accuracy
and trajectory validity, not just final output — with full execution tracing for observability.

### Project 5 — Probabilistic / Uncertainty (the "applied scientist" signal)

**Build:** A PyMC model on a problem where uncertainty matters (small-sample or risk-flavored).
Output distributions, not point estimates. Add conformal prediction (MAPIE) to a
separate predictive model for calibrated intervals.

**Tools shown:** PyMC, MAPIE, statistical rigor

**Resume claim:** Quantified uncertainty explicitly — probabilistic modeling and calibrated
prediction intervals — for accountable decision support.

### Project 6 — Fine-Tuning / Model Adaptation (research / model-building signal)

**Conditional:** worth building if the target role leans research or model-building. If the role
is platform/applied-engineering, skip it — serving (P1/P3) and agents (P4) already cover the LLM
story, and this becomes depth you won't be asked about.

**Build:** LoRA/QLoRA fine-tune of a small open model (e.g. a 1–8B instruct model) on a focused
task. The point is not the training loop — it's the **measured before/after eval**: define the
task, benchmark the base model, fine-tune, benchmark again, and report the delta honestly
(including where it *didn't* help). Track with MLflow. Keep the adapter small and the eval
rigorous — a clean before/after on a real task beats a big training run with no baseline.

**Tools shown:** PEFT/LoRA/QLoRA, HF Transformers + Accelerate, MLflow, task-specific eval harness

**Resume claim:** Adapted an open model to a target task with LoRA/QLoRA — quantified the gain
with a controlled before/after evaluation, not just a training run.

### Project 7 — Overhead Object Detection (CV anchor, merges with distributed)

**Domain:** geospatial / defense CV — broadens beyond cyber into the same national-security
mission space. Keep the framing analytic (what's in the image, situational awareness, HADR),
never targeting-flavored — the datasets below are all built and published for exactly that framing.

**Build:** Object detection on overhead imagery. Start on **DIOR** (23K pre-chipped 800×800
images, 20 classes) for a fast baseline, then optionally step up to **xView** (NGA/DIU benchmark,
~1M instances, 60 classes, 0.3m WorldView-3) for the credibility line. Fine-tune a detector
(YOLO, or Faster/Oriented R-CNN), evaluate with mAP. The applied-science differentiator is
handling the **small-object / large-scene** problem honestly — tiling / SAHI-style inference —
which is the actual hard part of overhead detection and separates a practitioner from a tutorial-runner.

**This is also your distributed-training artifact.** Overhead scenes are large and detection
training is heavy — a far better justification for going distributed (DDP → FSDP) than a tabular
model. P2's distributed work merges here rather than living as a separate project.

**License note:** overhead datasets vary — some permissive, several academic/non-commercial only
(iSAID, DOTA carry research-use terms). Note the license per dataset; "did you check the data
license" is a fair question in this space and knowing the answer is itself a signal.

**Tools shown:** object detection (YOLO / R-CNN), mAP evaluation, small-object tiling (SAHI),
geospatial imagery handling, PyTorch DDP → FSDP, MLflow

**Resume claim:** Built an overhead-imagery object detector on a defense benchmark (NGA/DIU xView),
handled the small-object/large-scene problem with tiled inference, and scaled training
single-GPU → distributed with a measured rationale.

### CV learning track (breadth, not portfolio-anchor depth)

Explicitly for *learning* range in CV, not as job-ready artifacts. Build these lighter than
Project 7 — enough to understand the technique and hold a conversation, reusing the same
geospatial domain so data-handling carries over. Be honest in framing: "built to learn," not
"polished portfolio piece." That honesty is more credible than claiming five deep CV projects.

- **Semantic / instance segmentation** — iSAID (masks over the same aerial domain, MS COCO
  format). Masks, not just boxes — a genuinely different capability from detection.
- **Change detection** — before/after imagery pairs. Huge in defense/geospatial and a distinctly
  different technique. High learning value.

Hard stop before **SAR (xView3 dark-vessel)**, **tracking**, and **video** — know they exist,
don't build yet. Add them later with evidence (a role that asks for them) rather than speculation.

---

## Subject matter — Domain: Network Threat Detection & Threat Intelligence

One domain through-line across all six projects, so they chain into a single
security-operations story instead of six disconnected demos. Telemetry datasets carry the
ML-heavy projects (P1/P2/P5/P6-eval); MITRE ATT&CK carries the document projects (P3/P4/P6).

**Scope note:** everything here is defensive / analytic — threat *detection*, telemetry
analysis, and threat-intelligence document work. No offensive tooling; that's neither
portfolio-buildable nor defensible in an interview.

### Datasets per project (2-3 options each)

**P1 — Spine (intrusion detection, XGBoost):**
- **NF-UNSW-NB15-v2** — NetFlow-standardized, ~2.4M flows, 9 attack subclasses, clean labels.
  Best default: manageable size, canonical benchmark.
- **CIC-IDS2017** — ~2.8M+ records, 80 features, 24 labels. The most widely-cited benchmark;
  interviewers know it.
- **NF-ToN-IoT-v2** — IoT/IIoT telemetry, larger and more imbalanced; good if you want an
  IoT/OT flavor.

*NetFlow-v2/v3 versions (UNSW-NB15, TON-IoT, CIC-IDS2018) share one common feature schema —
pick one for P1, and the same pipeline generalizes to the others.*

**P2 — Distributed training** *(now merges into Project 7 — CV detection is the better scale
justification; these tabular options remain valid only if you keep a separate cyber-domain
distributed artifact):*
- **CIC-IDS2017 full** (~5.6M rows × 80 features) — legitimate reason to go distributed, not a
  toy justification.
- **NF-ToN-IoT-v2** (~16.9M flows) — larger still if you want the scale story to be undeniable.
- **Packet-level NIDS data** (UNSW-NB15 + CIC-IDS2017, 230M+ packets via the `nids-datasets`
  package) — if you want to train on raw packet features, not just flows.

**P3 — RAG over threat intelligence:**
- **MITRE ATT&CK** (Enterprise) — public knowledge base of adversary tactics/techniques;
  ~200+ techniques, 450+ sub-techniques. The canonical CTI corpus; mirrors published patterns
  (RAGIntel, TechniqueRAG).
- **+ MITRE CVE / NVD descriptions** — add CVE records for a richer retrieval corpus.
- **+ CWE / D3FEND** — defensive-technique and weakness ontologies for cross-linking.

**P4 — Threat-investigation agent (tools over the P3 corpus):**
- ATT&CK retriever (from P3) as one tool + a **CVE lookup** tool (NVD API) + a
  **technique-mapping** tool (indicator → ATT&CK technique). Mirrors how real CTI pipelines
  link CVE ↔ ATT&CK ↔ CWE.

**P5 — Uncertainty on detection (conformal + Bayesian):**
- Reuse the **P1 intrusion dataset** — calibrated confidence on "is this flow malicious,"
  where false-positive cost is operational and class imbalance is real. Natural home for MAPIE.

**P6 — Fine-tuning (conditional):**
- **CTI-to-ATT&CK technique mapping** — LoRA-tune a small model to classify threat-report text
  into ATT&CK techniques. A documented task where domain fine-tuning beats zero-shot, so your
  before/after eval shows a real delta. (TRAM / SciBERT-style baselines exist to compare against.)

### Access notes
- UNSW datasets (UNSW-NB15, TON_IoT, BoT-IoT): research.unsw.edu.au project pages.
- NetFlow-standardized v2/v3: UQ NIDS datasets page (staff.itee.uq.edu.au/marius/NIDS_datasets).
- `nids-datasets` Python package: pip-installable, pulls curated parquet subsets.
- MITRE ATT&CK: available as STIX/JSON from the MITRE CTI GitHub repo — clean structured ingest.
- All are genuinely public; some UNSW pages gate behind a request form, so grab them early.

---

## Build order & sequencing

1. **Project 1 (Spine)** — do this first; it's the foundation and the highest-value single
   artifact. Serving and orchestration are the final two steps, not separate work.
2. **Project 3 (RAG + Ragas)** — second; your strongest public-sector differentiator and
   matches Databricks' agent momentum. Ends with a served endpoint.
3. **Project 4 (Agent)** — third, *right after RAG*, because it builds directly on the
   retriever while the code is fresh. This is the biggest 2026 addition.
4. **Project 5 (Bayesian/UQ)** — fourth; the depth signal. Expand only if it grabs you or the
   role is research-heavy.
5. **Project 6 (Fine-tuning)** — conditional. Build only if the role leans
   research/model-building; skip for platform/applied-engineering roles. Reuses the P3/P4 eval
   discipline, so it's cheap if you get there.
6. **Project 7 (CV detection + distributed)** — the modality expansion. Absorbs the old
   distributed-training project. **Sequence this AFTER shipping at least P1 and P3** — prove the
   finishing muscle on the domain you already know before opening a modality you don't.
7. **CV learning track (segmentation, change detection)** — last, and only after P7. Built to
   learn, not to anchor. Hard stop before SAR/tracking/video.

The single biggest risk to this plan is scoping replacing shipping. The order above front-loads
the highest-value cyber artifacts; everything CV waits until at least the Spine and the RAG app
exist and are defensible. A finished P1 teaches you more about your real appetite for P7 than any
amount of further planning.

In parallel (reading, cheap): build compliance literacy — FedRAMP, IL4/IL5, NIST AI RMF —
and practice connecting each to a platform governance feature.

At the end: earn a Databricks certification (ML Associate/Professional or Data Engineer
Associate) to make the whole thing legible on paper.

---

## Where to run it (all free)

Build everything on open source first — `pip install mlflow delta-spark pyspark`,
plus FAISS/Chroma, LangChain, Ragas, PyMC, LangGraph — running locally or in Docker.
This proves you understand what the managed platform does for you, which interviewers respect more.
Then layer the free Databricks tier on top as a quick "and I know the managed platform" demonstration.
Serving and orchestration are where the OSS-first / managed contrast is sharpest — do it by hand
once (FastAPI + Docker + Airflow), then show the native equivalent (Model Serving + Workflows).

---

## The boundary

If you find yourself researching a new tool before Project 1 exists, that's the collecting trap.
The list above is deliberately closed. Ship the spine, then the RAG app, then the agent, then
decide if you even need the rest.

Hold one line specifically: **one serving stack, one orchestrator, one agent framework.**
The additions here close the deploy → run → observe arc; they are not permission to collect a
second of anything.

A finished project beats ten named tools.
