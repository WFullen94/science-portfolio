"""Airflow DAG — scheduled NIDS pipeline with a drift-triggered retraining path.

    ingest -> drift_check -> [drift?] --yes--> retrain_features -> retrain_train -> done
                                     --no---> skip_retrain --------------------------^

Each stage runs as a BashOperator invoking `python -m spine.<stage>` in the
project venv, so this DAG file imports only stdlib + Airflow and parses without
the heavy data stack (Spark/MLflow live on the worker, not the scheduler).

Deploy: symlink/copy this file into your AIRFLOW_HOME/dags and set the two env
vars below (SPINE_PROJECT_DIR, SPINE_PYTHON) for your environment.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator

# Where the project + its venv live (override via env for a real deployment).
PROJECT_DIR = os.environ.get(
    "SPINE_PROJECT_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
PYTHON = os.environ.get("SPINE_PYTHON", os.path.join(PROJECT_DIR, ".venv", "bin", "python"))
DRIFT_RESULT = os.path.join(PROJECT_DIR, "data", "reports", "drift_result.json")

# Task ids the branch chooses between (mirror spine.drift constants).
RETRAIN_TASK = "retrain_features"
SKIP_TASK = "skip_retrain"


def _stage(module: str) -> str:
    return f"cd {PROJECT_DIR} && {PYTHON} -m {module}"


def _decide_branch(**_) -> str:
    """Read the drift result written by spine.drift; pick the next task."""
    if not os.path.exists(DRIFT_RESULT):
        return SKIP_TASK
    with open(DRIFT_RESULT) as fh:
        return json.load(fh).get("decision", SKIP_TASK)


default_args = {
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="spine_nids_pipeline",
    description="NetFlow intrusion-detection pipeline with drift-triggered retraining",
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["nids", "spine", "mlops"],
) as dag:
    ingest = BashOperator(task_id="ingest", bash_command=_stage("spine.ingest"))
    drift_check = BashOperator(task_id="drift_check", bash_command=_stage("spine.drift"))
    drift_branch = BranchPythonOperator(task_id="drift_branch", python_callable=_decide_branch)

    retrain_features = BashOperator(task_id=RETRAIN_TASK, bash_command=_stage("spine.features"))
    retrain_train = BashOperator(task_id="retrain_train", bash_command=_stage("spine.train"))
    skip_retrain = EmptyOperator(task_id=SKIP_TASK)

    # Runs on either branch (retrain path or skip path).
    done = EmptyOperator(task_id="done", trigger_rule="none_failed_min_one_success")

    ingest >> drift_check >> drift_branch >> [retrain_features, skip_retrain]
    retrain_features >> retrain_train >> done
    skip_retrain >> done
