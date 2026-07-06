# Project 1 — The Spine. Run stages in order, or `make all`.
# Assumes the venv is active (scripts/setup_env.sh) so JAVA_HOME + deps are set.

.PHONY: setup ingest features train serve drift all clean test

setup:          ## Create venv, install deps, wire JAVA_HOME
	bash scripts/setup_env.sh

ingest:         ## Download NetFlow data -> bronze Delta table + validate
	python -m spine.ingest

features:       ## PySpark feature engineering: silver -> gold train/test
	python -m spine.features

train:          ## XGBoost + Optuna + MLflow + SHAP, register best model
	python -m spine.train

serve:          ## Serve the registered model (FastAPI on :8000)
	uvicorn spine.serve:app --host 0.0.0.0 --port 8000

drift:          ## Evidently drift check -> emit retrain signal
	python -m spine.drift

all: ingest features train  ## Full offline pipeline (data -> registered model)

test:
	pytest -q

clean:          ## Remove local data + mlflow runs (keeps code)
	rm -rf data/delta data/reports data/great_expectations data/spark-warehouse mlruns
