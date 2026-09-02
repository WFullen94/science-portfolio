"""Stage 2 — Train: fine-tune a CNN backbone as a real/fake face classifier.

Two-stage fine-tune: warm up the new classifier head with the backbone frozen for
`freeze_backbone_epochs`, then unfreeze the whole network — the standard transfer-
learning recipe (an untrained head would otherwise send large, destructive gradients
through the pretrained backbone in the first steps). Early-stops on validation ROC-AUC.

`backbone` is an explicit argument (default: cfg["model"]["backbone"]) so compare.py
can train several architectures on the IDENTICAL data split (list_items/
stratified_split are deterministic given the same seed) — same split, different model,
a fair comparison. Each backbone gets its own checkpoint subdirectory.
"""

from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dfdetect.config import load_config, resolve
from dfdetect.data import list_items, prepare, stratified_split
from dfdetect.dataset import FaceDataset
from dfdetect.metrics import binary_metrics
from dfdetect.modeling import BACKBONES, build_model, pick_device, set_backbone_trainable


@torch.no_grad()
def _scores(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, scores = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x).squeeze(-1)
        scores.append(torch.sigmoid(logits).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(scores)


def run(cfg=None, backbone: str | None = None, log_to_mlflow: bool = True) -> dict:
    cfg = cfg or load_config()
    mcfg, tcfg = cfg["model"], cfg["train"]
    backbone = backbone or mcfg["backbone"]
    device = pick_device(mcfg["device"])
    torch.manual_seed(tcfg["random_state"])

    paths = prepare(cfg)
    items = list_items(paths["pool"], cfg["dataset"]["max_per_class"])
    train_items, val_items = stratified_split(items, cfg["split"]["val_fraction"],
                                              cfg["split"]["random_state"])
    print(f"[train] backbone={backbone}  pool split: train={len(train_items)} val={len(val_items)}")

    train_ds = FaceDataset(train_items, mcfg["img_size"], augment=True)
    val_ds = FaceDataset(val_items, mcfg["img_size"], augment=False)
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False)

    model = build_model(backbone).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])

    best_auc, best_state, patience = -1.0, None, 0
    for epoch in range(1, tcfg["epochs"] + 1):
        set_backbone_trainable(model, trainable=epoch > tcfg["freeze_backbone_epochs"])
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x).squeeze(-1), y)
            loss.backward()
            opt.step()
            running += loss.item()

        y_val, s_val = _scores(model, val_loader, device)
        val_metrics = binary_metrics(y_val, s_val, cfg["eval"]["threshold"])
        frozen = "frozen" if epoch <= tcfg["freeze_backbone_epochs"] else "unfrozen"
        print(f"[train] epoch {epoch:02d} ({frozen})  train_loss={running/len(train_loader):.4f}  "
              f"val ROC-AUC={val_metrics['roc_auc']:.4f}  val_acc={val_metrics['accuracy']:.4f}")

        if val_metrics["roc_auc"] > best_auc:
            best_auc, patience = val_metrics["roc_auc"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= tcfg["patience"]:
                print(f"[train] early stop at epoch {epoch} (best val AUC={best_auc:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt = resolve(tcfg["ckpt_dir"]) / backbone
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt / "best.pt")
    print(f"[train] saved best checkpoint -> {ckpt / 'best.pt'}")

    if log_to_mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
            mlflow.set_experiment(cfg["mlflow"]["experiment"])
            with mlflow.start_run(run_name=f"{backbone}-finetune"):
                mlflow.log_params({"backbone": backbone, "epochs": tcfg["epochs"],
                                   "lr": tcfg["lr"], "batch_size": tcfg["batch_size"]})
                mlflow.log_metric("best_val_roc_auc", best_auc)
        except Exception as exc:
            print(f"[train] mlflow logging skipped: {exc}")

    result = {"backbone": backbone, "best_val_roc_auc": best_auc, "checkpoint": str(ckpt / "best.pt")}
    (resolve("data/reports")).mkdir(parents=True, exist_ok=True)
    (resolve("data/reports") / f"train_result_{backbone}.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default=None, choices=BACKBONES)
    args = ap.parse_args()
    run(backbone=args.backbone)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
