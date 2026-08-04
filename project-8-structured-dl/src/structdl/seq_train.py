"""Shared training loop for the stage-2 sequence models.

Both the time-series Transformer and the GRU baseline are pure-torch binary
sequence classifiers with the same recipe: BCE with a positive-class weight,
early-stop on validation ROC-AUC, evaluate once on the held-out test set, log to
MLflow. Factoring it here keeps the two model files to just their architecture,
and guarantees the head-to-head differs only in the model — same data, same loop.
"""

from __future__ import annotations

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from structdl.metrics import binary_metrics
from structdl.sequence_data import SeqBundle


def pick_device(requested) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def _scores(model, X, device, batch_size) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).float().to(device)
        out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out)


def train_sequence_model(model, b: SeqBundle, mcfg, run_name: str,
                         experiment: str, tracking_uri: str,
                         log_to_mlflow: bool = True) -> dict:
    """Train `model` on the sequence bundle; return held-out test metrics.

    `mcfg` is the model's config block (lr, batch_size, max_epochs, patience, ...).
    """
    torch.manual_seed(mcfg["random_state"])
    device = pick_device(mcfg["device"])
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{run_name}] device={device}  parameters={n_params:,}")

    pos = float((b.ytrain == 1).sum())
    neg = float((b.ytrain == 0).sum())
    pos_weight = torch.tensor([neg / pos if pos else 1.0], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=mcfg["lr"],
                            weight_decay=mcfg["weight_decay"])
    train_loader = _loader(b.Xtrain, b.ytrain, mcfg["batch_size"], shuffle=True)

    best_auc, best_state, patience = -1.0, None, 0
    for epoch in range(1, mcfg["max_epochs"] + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        val_auc = binary_metrics(b.yval, _scores(model, b.Xval, device,
                                                 mcfg["batch_size"]))["roc_auc"]
        print(f"[{run_name}] epoch {epoch:02d}  val ROC-AUC={val_auc:.4f}")
        if val_auc > best_auc:
            best_auc, patience = val_auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= mcfg["patience"]:
                print(f"[{run_name}] early stop at epoch {epoch} (best val AUC={best_auc:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = binary_metrics(b.ytest, _scores(model, b.Xtest, device, mcfg["batch_size"]))
    print(f"[{run_name}] test ROC-AUC={metrics['roc_auc']:.4f} "
          f"PR-AUC={metrics['pr_auc']:.4f} F1={metrics['f1']:.4f}")

    if log_to_mlflow:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("model", run_name)
            mlflow.log_params({k: mcfg[k] for k in ("lr", "batch_size", "max_epochs")})
            mlflow.log_param("n_params", n_params)
            mlflow.log_metric("best_val_roc_auc", best_auc)
            mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
    return metrics
