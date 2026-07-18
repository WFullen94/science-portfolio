# Project 1 — The Spine

End-to-end ML on a lakehouse pattern, applied to **network intrusion detection**.
The full lifecycle in one defensible artifact:

```
download → Delta (bronze) → validate → PySpark features → Delta (gold)
        → train (XGBoost + Optuna) → MLflow track + register → SHAP explain
        → serve (FastAPI) → orchestrate (Airflow) → drift-triggered retrain
```

**Dataset:** NF-UNSW-NB15-v2 (NetFlow-v2 standardized, ~2.4M flows, 9 attack subclasses).
The v2 schema is shared across NF-ToN-IoT-v2 / NF-CSE-CIC-IDS2018, so this pipeline generalizes
to those by changing only the `dataset` block in [conf/config.yaml](conf/config.yaml).

## Stack (roadmap "lead with" tools, each earning its place)

| Stage | Tool |
|-------|------|
| Storage | **Delta Lake** — versioned, ACID bronze/silver/gold tables |
| Processing | **PySpark** — distributed feature engineering |
| Validation | **Great Expectations** — input data-quality gate |
| Modeling | **XGBoost** + **scikit-learn** |
| Tuning | **Optuna** — hyperparameter search |
| Tracking/Registry | **MLflow** — autolog, model registry, aliases |
| Explainability | **SHAP** |
| Serving | **FastAPI** (+ Dockerfile) — real-time endpoint |
| Orchestration | **Airflow** — scheduled DAG |
| Drift/Monitoring | **Evidently** — drift check wired to the retrain trigger |

## Quickstart

```bash
cd project-1-spine
make setup                 # venv + deps + JAVA_HOME (needs: brew install openjdk@17)
source .venv/bin/activate
make all                   # ingest -> features -> train  (data -> registered model)
make serve                 # real-time endpoint on :8000
```

## Layout

```
conf/config.yaml     single source of truth (paths, dataset, model, thresholds)
src/spine/
  config.py          config loader + Delta-enabled Spark session
  ingest.py          download + land raw flows into bronze Delta        [commit 2]
  validate.py        Great Expectations suite on the input              [commit 2]
  features.py        PySpark feature engineering -> gold Delta          [commit 3]
  train.py           XGBoost + Optuna + MLflow + SHAP + register        [commit 4]
  serve.py           FastAPI real-time scoring endpoint                 [commit 5]
  drift.py           Evidently drift check -> retrain signal            [commit 6]
dags/                Airflow DAG                                        [commit 6]
docker/              Dockerfile for the serving image                   [commit 5]
```

## Serving

The endpoint loads the registered model **by alias** (`@staging`, else `@candidate`,
else latest) and scores raw NetFlow-v2 flows — the request carries raw flow counters,
the server applies the same feature engineering used in training (a pandas twin of the
Spark path, guarded by a parity test) and returns a maliciousness probability + verdict.

```bash
make serve                 # uvicorn on :8000, loads model from the local registry
curl -s localhost:8000/health
curl -s localhost:8000/predict -H 'content-type: application/json' \
  -d '{"flows":[{"IN_BYTES":4200,"IN_PKTS":48,"OUT_BYTES":300,"OUT_PKTS":4,"PROTOCOL":6,"TCP_FLAGS":22,"FLOW_DURATION_MILLISECONDS":12}]}'
```

Interactive docs at `/docs`. Verified on held-out data: **16/16 real flows** classified
correctly (attacks ≈0.998, benign <0.08).

**Containerized serving** bakes the promoted model into the image (no MLflow server needed
at runtime — `serve.py` loads it from `SPINE_MODEL_PATH`):

```bash
make docker-build          # export @staging model -> serving/model, then docker build
make docker-run            # serve the container on :8000
```

## Managed-platform equivalents

Built OSS-first to show the mechanics; the Databricks-native mapping:
Delta tables → Unity Catalog · MLflow → managed MLflow · FastAPI → Model Serving ·
Airflow → Databricks Workflows · Great Expectations/Evidently → Lakehouse Monitoring.
