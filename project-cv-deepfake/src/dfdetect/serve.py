"""Stage 4b — Serve the ONNX-exported classifier behind FastAPI.

Torch-free: inference runs on ONNX Runtime, so the serving container never installs
torch/torchvision (see requirements-serve.txt / docker/Dockerfile) — a much smaller,
faster-to-build image than the training environment.
"""

from __future__ import annotations

import io
from functools import lru_cache

import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image

from dfdetect.config import load_config, resolve

app = FastAPI(title="dfdetect — deepfake / synthetic-face detection")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@lru_cache(maxsize=1)
def _load():
    import onnxruntime as ort
    cfg = load_config()
    onnx_path = resolve(cfg["serve"]["onnx"])
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return cfg, session


def _preprocess(img: Image.Image, size: int) -> np.ndarray:
    img = img.convert("RGB").resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(2, 0, 1)[None, ...]  # (1, 3, H, W)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@app.get("/")
def root():
    cfg, _ = _load()
    return {"service": "dfdetect", "model": cfg["model"]["backbone"], "runtime": "onnxruntime"}


@app.get("/health")
def health():
    _load()
    return {"status": "ok"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    cfg, session = _load()
    img = Image.open(io.BytesIO(await image.read()))
    x = _preprocess(img, cfg["model"]["img_size"])
    logit = session.run(None, {"image": x.astype(np.float32)})[0]
    fake_prob = float(_sigmoid(logit)[0, 0])
    return {"fake_probability": round(fake_prob, 4),
            "prediction": "fake" if fake_prob >= cfg["eval"]["threshold"] else "real"}
