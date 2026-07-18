"""Stage 4 — Serve: real-time NIDS scoring behind a FastAPI endpoint.

Loads the registered model by alias (@staging, else @candidate, else latest) at
startup, and scores raw NetFlow-v2 flows: the request carries raw flow counters,
the server applies the same feature engineering used in training (pandas twin),
then returns a maliciousness probability + verdict.

Run:  uvicorn spine.serve:app --port 8000     (or: make serve)
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

from spine import schema
from spine.config import load_config
from spine.features import MODEL_FEATURES, engineer_pandas, to_model_matrix

ALIAS_PREFERENCE = ("staging", "candidate")
THRESHOLD = 0.5


class _Bundle:
    """Holds the loaded model + provenance."""
    model = None
    name: str = ""
    version: str = ""
    alias: str = ""


bundle = _Bundle()


def _resolve_version(client: MlflowClient, name: str):
    for alias in ALIAS_PREFERENCE:
        try:
            mv = client.get_model_version_by_alias(name, alias)
            return mv.version, alias
        except Exception:
            continue
    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        raise RuntimeError(f"No versions registered for model '{name}'")
    latest = max(versions, key=lambda v: int(v.version))
    return latest.version, "latest"


def _load_baked(model_path: Path) -> bool:
    """Load a model baked into the image (SPINE_MODEL_PATH), decoupling the
    container from any MLflow server. Returns True if a baked model was loaded."""
    if not model_path.exists():
        return False
    bundle.model = mlflow.xgboost.load_model(str(model_path))
    meta_file = model_path / "spine_meta.json"
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    bundle.name = meta.get("name", "spine-nids-detector")
    bundle.version = str(meta.get("version", "baked"))
    bundle.alias = meta.get("alias", "baked")
    print(f"[serve] loaded baked model from {model_path} "
          f"({bundle.name} v{bundle.version} @{bundle.alias})")
    return True


def load_model() -> None:
    # Container / offline path: a model baked into the image at SPINE_MODEL_PATH.
    baked = os.environ.get("SPINE_MODEL_PATH")
    if baked and _load_baked(Path(baked)):
        return

    # Dev path: resolve from the local MLflow registry by alias.
    cfg = load_config()
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    name = cfg["mlflow"]["registered_model"]
    client = MlflowClient()
    version, alias = _resolve_version(client, name)
    bundle.model = mlflow.xgboost.load_model(f"models:/{name}/{version}")
    bundle.name, bundle.version, bundle.alias = name, str(version), alias
    print(f"[serve] loaded {name} v{version} (@{alias})")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Spine NIDS Detector",
    description="Real-time network-intrusion detection on NetFlow-v2 flows.",
    version="1.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- schemas
class PredictRequest(BaseModel):
    flows: list[dict[str, float]] = Field(
        ...,
        description="Raw NetFlow-v2 flow records (numeric features). "
        "Missing features default to 0; identifiers are ignored.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {"flows": [{
                "IN_BYTES": 4200, "IN_PKTS": 48, "OUT_BYTES": 300, "OUT_PKTS": 4,
                "PROTOCOL": 6, "L7_PROTO": 0, "TCP_FLAGS": 22,
                "FLOW_DURATION_MILLISECONDS": 12, "MIN_TTL": 5, "MAX_TTL": 64,
                "MAX_IP_PKT_LEN": 1500, "MIN_IP_PKT_LEN": 40,
            }]}
        }
    }


class Prediction(BaseModel):
    malicious_probability: float
    prediction: int
    verdict: str


class PredictResponse(BaseModel):
    # 'model_' is a pydantic-protected namespace; opt out for these field names.
    model_config = {"protected_namespaces": ()}
    model_name: str
    model_version: str
    model_alias: str
    predictions: list[Prediction]


# ------------------------------------------------------------------------- endpoints
@app.get("/")
def root():
    return {"service": "spine-nids-detector", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {
        "status": "ok" if bundle.model is not None else "no_model",
        "model_name": bundle.name,
        "model_version": bundle.version,
        "model_alias": bundle.alias,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if bundle.model is None:
        raise HTTPException(503, "model not loaded")
    if not req.flows:
        raise HTTPException(422, "no flows provided")

    raw = pd.DataFrame(req.flows)
    # Ensure every raw NetFlow feature exists (missing -> 0) before engineering.
    for col in schema.NUMERIC_FEATURES:
        if col not in raw.columns:
            raw[col] = 0.0
    raw = raw.fillna(0.0)

    engineered = engineer_pandas(raw)
    X = to_model_matrix(engineered, MODEL_FEATURES)
    proba = bundle.model.predict_proba(X)[:, 1]

    preds = [
        Prediction(
            malicious_probability=round(float(p), 6),
            prediction=int(p >= THRESHOLD),
            verdict="attack" if p >= THRESHOLD else "benign",
        )
        for p in proba
    ]
    return PredictResponse(
        model_name=bundle.name,
        model_version=bundle.version,
        model_alias=bundle.alias,
        predictions=preds,
    )
