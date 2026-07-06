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

## Managed-platform equivalents

Built OSS-first to show the mechanics; the Databricks-native mapping:
Delta tables → Unity Catalog · MLflow → managed MLflow · FastAPI → Model Serving ·
Airflow → Databricks Workflows · Great Expectations/Evidently → Lakehouse Monitoring.
