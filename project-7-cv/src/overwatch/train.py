"""Stage 2 — Fine-tune YOLOv8n on VisDrone.

Starts from COCO-pretrained `yolov8n.pt` and fine-tunes on VisDrone's 10 aerial
classes. Small objects dominate, so we train at a larger input size (imgsz 960).
Ultralytics owns the training loop; we drive it from config and log the final
validation metrics (mAP@50, mAP@50-95) to MLflow.

Distributed note: ultralytics runs multi-GPU DDP by passing `device=0,1,2,...` —
the same fine-tune scales to a multi-GPU box unchanged. Here it's single-device (MPS).
"""

from __future__ import annotations

import argparse

from overwatch.config import load_config, resolve
from overwatch.utils import pick_device


def run(cfg=None, data_yaml: str | None = None, epochs: int | None = None,
        imgsz: int | None = None, fraction: float = 1.0) -> dict:
    from ultralytics import YOLO

    cfg = cfg or load_config()
    mcfg, tcfg = cfg["model"], cfg["train"]
    device = pick_device(mcfg["device"])
    data = data_yaml or str(resolve(cfg["dataset"]["yaml"]))
    epochs = epochs if epochs is not None else tcfg["epochs"]
    imgsz = imgsz if imgsz is not None else mcfg["imgsz"]

    print(f"[train] fine-tuning {mcfg['weights']} on {data} | epochs={epochs} "
          f"imgsz={imgsz} fraction={fraction} device={device}")
    model = YOLO(mcfg["weights"])
    results = model.train(
        data=data, epochs=epochs, imgsz=imgsz, batch=tcfg["batch"], fraction=fraction,
        device=device, seed=tcfg["seed"], patience=tcfg["patience"],
        project=str(resolve(tcfg["runs_dir"])), name="finetune", exist_ok=True,
        verbose=True,
    )

    rd = results.results_dict
    metrics = {
        "mAP50": float(rd.get("metrics/mAP50(B)", 0.0)),
        "mAP50_95": float(rd.get("metrics/mAP50-95(B)", 0.0)),
        "precision": float(rd.get("metrics/precision(B)", 0.0)),
        "recall": float(rd.get("metrics/recall(B)", 0.0)),
    }
    best = resolve(tcfg["runs_dir"]) / "finetune" / "weights" / "best.pt"
    print(f"[train] done — mAP@50={metrics['mAP50']:.4f} "
          f"mAP@50-95={metrics['mAP50_95']:.4f} | best: {best}")

    try:
        import mlflow
        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        mlflow.set_experiment(cfg["mlflow"]["experiment"])
        with mlflow.start_run(run_name="yolov8n-finetune"):
            mlflow.log_params({"weights": mcfg["weights"], "epochs": epochs,
                               "imgsz": mcfg["imgsz"], "batch": tcfg["batch"]})
            mlflow.log_metrics(metrics)
    except Exception as exc:  # tracking is a nicety, not a hard dep of training
        print(f"[train] mlflow logging skipped: {exc}")

    return {"metrics": metrics, "weights": str(best)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="dataset yaml (default: full VisDrone)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="fraction of train images to use (speed knob on a GPU-less box)")
    args = ap.parse_args()
    run(data_yaml=args.data, epochs=args.epochs, imgsz=args.imgsz, fraction=args.fraction)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
