# Overwatch — Overhead Object Detection on VisDrone, with SAHI Tiling

An end-to-end **overhead (drone/aerial) object-detection service**: fine-tune YOLOv8 on
**VisDrone**, then answer the question that actually matters for aerial imagery — *how much does
**slicing-aided (SAHI) tiled inference** recover on tiny objects, and what does it cost?* — with a
controlled head-to-head. Then export to ONNX and serve it behind FastAPI.

Same DNA as the rest of the portfolio: a **fair, honest comparison** (one model, one mAP
implementation scoring both paths) rather than a single number, and a real **data → train → eval →
export → serve → deploy** pipeline, not a notebook.

## Why VisDrone + SAHI

VisDrone is drone imagery — 10 classes (pedestrian, car, van, bus, …), and objects are **tiny**
(often a handful of pixels). A detector run once on the whole downscaled frame simply can't see many
of them. **SAHI** (Slicing-Aided Hyper Inference) slices each frame into overlapping tiles, runs the
detector per tile, and merges the results — trading extra forward passes (latency) for small-object
recall. Whether that trade is worth it is an empirical question this project answers.

## The head-to-head — untiled vs SAHI-tiled

Fine-tuned YOLOv8n, 200 VisDrone val images. Both paths use the **same** model and the **same**
mAP@50 implementation ([metrics.py](src/overwatch/metrics.py)) so the comparison is fair:

| method | mAP@50 | small-obj recall | ms/img |
|---|---|---|---|
| **Untiled** | 0.1528 | 0.3035 | 16 |
| **SAHI-tiled** | **0.1746** | **0.3588** | 101 |
| **Δ (SAHI − untiled)** | **+0.0217** | **+0.0553** | **×6.5** |

**SAHI earns its keep — at a price.** Tiling lifts mAP@50 by **+0.022 (~14% relative)** and
small-object recall by **+0.055 (~18% relative)** — recovering exactly the tiny objects a single
downscaled pass misses — but costs **6.5× the latency** (16 → 101 ms/img). That's the decision on the
table: for offline/forensic aerial analysis the recall is worth it; for a real-time feed you'd weigh
the tiling cost or tile only regions of interest. (`small-obj recall` = recall on GT boxes < 32×32 px,
SAHI's target regime. Numbers from a short demonstration fine-tune — 8 epochs on 25% of VisDrone at
imgsz 640; full-dataset training lifts both rows, and the *gap* is the point.)

## Pipeline

```bash
make setup     # venv + deps
make verify    # confirm the VisDrone YOLO dataset resolves; build a tiny CI sample
make train     # fine-tune YOLOv8n on VisDrone (MLflow-tracked)
make eval      # untiled vs SAHI-tiled head-to-head -> data/reports/sahi_headtohead.json
make export    # fine-tuned model -> ONNX (data/overwatch.onnx)
make serve     # FastAPI /predict (tiles at inference when serve.use_sahi)
make test      # mAP@50 metric tests (pure numpy — no data/model needed)
```

Small-object detection needs a large input, so training/eval run at **imgsz 960** by default. On a
GPU-less box, `make train` uses a subset for a tractable demonstration fine-tune
(`--fraction`/`--epochs`/`--imgsz` knobs); the same command trains the full set on a GPU, and
ultralytics runs **multi-GPU DDP** just by passing `device=0,1,2,3`.

## Serving

`POST /predict` takes an uploaded image and returns detections (`box`, `score`, `class`); it tiles
with SAHI when `serve.use_sahi` is set, matching the eval-time winner. The
[Dockerfile](docker/Dockerfile) bakes code + config; mount trained weights as a volume
(`-v $(pwd)/data:/app/data`) or it falls back to pretrained.

```bash
curl -F image=@frame.jpg http://localhost:8000/predict
```

## The data

VisDrone-DET (6,471 train / 548 val, 10 classes), already materialized in YOLO format (images
symlinked into the raw 1.6GB tree) in the computer-vision repo — **referenced, not copied**.
`make verify` confirms reachability and carves a small bundled sample so CI/tests never need the
full tree. Reproduce elsewhere via the public download URLs in [conf/config.yaml](conf/config.yaml).

## Layout

```
conf/config.yaml            single source of truth (dataset, model, train, eval/SAHI, serve, MLflow)
src/overwatch/
  data.py                   verify VisDrone dataset + build CI sample                    [stage 1]
  train.py                  fine-tune YOLOv8n (ultralytics + MLflow)                      [stage 2]
  metrics.py                self-contained mAP@50 (scores BOTH paths — fair)
  evaluate.py               untiled vs SAHI-tiled head-to-head (mAP, small-recall, ms)    [stage 3]
  export_onnx.py            fine-tuned model -> ONNX                                       [stage 4a]
  serve.py                  FastAPI /predict (optional SAHI tiling)                        [stage 4b]
docker/Dockerfile           containerized detection service
tests/                      mAP@50 metric correctness (pure numpy, CI-fast)
```

## The interview framing

> "I built an overhead-detection service on VisDrone — fine-tuned YOLOv8, then ran a controlled
> untiled-vs-SAHI-tiled head-to-head scored by one mAP implementation, so the small-object recall
> gain and its latency cost are both on the table. Then ONNX export + a FastAPI service that tiles at
> inference. The point isn't 'tiling is better' — it's quantifying the recall/latency trade so you
> can decide whether SAHI earns its cost for a given deployment."

## Extensions
A Faster R-CNN baseline for a YOLO-vs-R-CNN architecture comparison; full-dataset training on a GPU
box; TensorRT export for lower-latency serving; oriented bounding boxes (VisDrone vehicles).
