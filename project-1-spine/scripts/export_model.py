"""Export the promoted model from the MLflow registry into a portable directory.

Produces serving/model/ — a self-contained MLflow xgboost model plus a small
spine_meta.json — so the serving container can be built with the model baked in,
independent of any MLflow server. Run before `docker build`.

    python scripts/export_model.py            # exports the @staging model
"""

from __future__ import annotations

import json
import shutil

import mlflow
from mlflow.tracking import MlflowClient

from spine.config import load_config, resolve
from spine.serve import ALIAS_PREFERENCE, _resolve_version

OUT_DIR = resolve("serving/model")


def main() -> int:
    cfg = load_config()
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    name = cfg["mlflow"]["registered_model"]
    client = MlflowClient()
    version, alias = _resolve_version(client, name)

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.parent.mkdir(parents=True, exist_ok=True)

    local = mlflow.artifacts.download_artifacts(f"models:/{name}/{version}")
    shutil.copytree(local, OUT_DIR)
    (OUT_DIR / "spine_meta.json").write_text(
        json.dumps({"name": name, "version": str(version), "alias": alias}, indent=2)
    )
    print(f"[export] {name} v{version} (@{alias}) -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
