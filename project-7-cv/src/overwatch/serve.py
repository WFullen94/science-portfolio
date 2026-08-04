"""Stage 4b — Serve the detector behind a FastAPI /predict endpoint.

Loads the fine-tuned YOLOv8 model once at startup (lru_cache). `/predict` takes an
uploaded image and returns detections; with `serve.use_sahi` it tiles the frame
(SAHI) for small-object recall, matching the eval-time winner. Endpoints: `/`,
`/health`, `POST /predict`.
"""

from __future__ import annotations

import io
from functools import lru_cache

import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image

from overwatch.config import load_config, resolve
from overwatch.evaluate import _weights
from overwatch.utils import pick_device

app = FastAPI(title="Overwatch — overhead object detection")


@lru_cache(maxsize=1)
def _load():
    cfg = load_config()
    device = pick_device(cfg["model"]["device"])
    weights = _weights(cfg)
    from ultralytics import YOLO
    model = YOLO(weights)
    det_model = None
    if cfg["serve"]["use_sahi"]:
        from sahi import AutoDetectionModel
        det_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path=weights,
            confidence_threshold=cfg["serve"]["conf"],
            device="cuda:0" if device == "0" else device,
        )
    return cfg, model, det_model, device


def _detect(image: Image.Image) -> list[dict]:
    cfg, model, det_model, device = _load()
    names = cfg["dataset"]["names"]
    dets = []
    if det_model is not None:  # SAHI tiled path
        from sahi.predict import get_sliced_prediction
        s = cfg["eval"]["sahi"]
        res = get_sliced_prediction(
            np.array(image), det_model,
            slice_height=s["slice_height"], slice_width=s["slice_width"],
            overlap_height_ratio=s["overlap_ratio"], overlap_width_ratio=s["overlap_ratio"],
            verbose=0,
        )
        for op in res.object_prediction_list:
            dets.append({"box": [round(float(v), 1) for v in op.bbox.to_xyxy()],
                         "score": round(float(op.score.value), 4),
                         "class_id": int(op.category.id), "class": str(op.category.name)})
    else:  # plain untiled path
        r = model.predict(np.array(image), conf=cfg["serve"]["conf"],
                          imgsz=cfg["model"]["imgsz"], device=device, verbose=False)[0]
        for b in (r.boxes or []):
            cid = int(b.cls.item())
            dets.append({"box": [round(v, 1) for v in b.xyxy[0].tolist()],
                         "score": round(float(b.conf.item()), 4),
                         "class_id": cid, "class": names[cid]})
    return dets


@app.get("/")
def root():
    cfg, _, det_model, device = _load()
    return {"service": "overwatch", "dataset": cfg["dataset"]["name"],
            "sahi": det_model is not None, "device": device}


@app.get("/health")
def health():
    _load()
    return {"status": "ok"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await image.read())).convert("RGB")
    dets = _detect(img)
    return {"width": img.width, "height": img.height,
            "n_detections": len(dets), "detections": dets}
