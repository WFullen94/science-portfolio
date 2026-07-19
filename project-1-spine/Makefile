# Project 1 — The Spine. Run stages in order, or `make all`.
# Assumes the venv is active (scripts/setup_env.sh) so JAVA_HOME + deps are set.

.PHONY: setup ingest features train serve drift all clean test export-model docker-build docker-run

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

export-model:   ## Export the promoted model into serving/model for baking
	python scripts/export_model.py

docker-build: export-model  ## Build the serving image (model baked in)
	docker build -f docker/Dockerfile -t spine-nids-serve .

docker-run:     ## Run the serving container on :8000
	docker run --rm -p 8000:8000 spine-nids-serve

dag-validate:   ## Parse the Airflow DAG (zero import errors) in the airflow venv
	AIRFLOW__CORE__LOAD_EXAMPLES=False .venv-airflow/bin/python -c \
	  "from airflow.models import DagBag; d=DagBag('dags',include_examples=False); \
	   print('import errors:', d.import_errors or 'none'); assert not d.import_errors"

all: ingest features train  ## Full offline pipeline (data -> registered model)

test:
	pytest -q

clean:          ## Remove local data + mlflow runs (keeps code)
	rm -rf data/delta data/reports data/great_expectations data/spark-warehouse mlruns
