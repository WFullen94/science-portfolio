"""Stage 1b — the challenger: a hand-rolled FT-Transformer.

FT-Transformer (Gorishniy et al., 2021, "Revisiting Deep Learning Models for
Tabular Data") applies a Transformer to tabular rows. The trick is the
**feature tokenizer**: every feature becomes one d_token embedding —

  * a numeric feature x_i  ->  x_i * W_i + b_i        (a learned per-feature vector)
  * a categorical value    ->  Embedding[value]        (a learned per-category vector)

Stack those tokens, prepend a learned [CLS] token, run a Transformer encoder so
features attend to each other, and read the classification off the final [CLS].
That per-feature attention is exactly what a tree can't do and what pure-numeric
data can't exercise — hence the categorical UNSW-NB15 dataset.

I hand-roll the tokenizer and CLS pooling (the tabular-specific parts) and use
torch's own encoder layer for the well-trodden attention/FFN backbone.
"""

from __future__ import annotations

import json

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from structdl.config import load_config, resolve
from structdl.data import DataBundle, build_bundle
from structdl.metrics import binary_metrics


def _pick_device(requested) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FeatureTokenizer(nn.Module):
    """Raw features -> (batch, n_features, d_token) tokens. The tabular-specific bit."""

    def __init__(self, n_num: int, cat_cardinalities: list[int], d_token: int):
        super().__init__()
        self.n_num = n_num
        # Numeric: a learned weight + bias vector per feature (affine to d_token).
        self.num_weight = nn.Parameter(torch.empty(n_num, d_token))
        self.num_bias = nn.Parameter(torch.empty(n_num, d_token))
        # Categorical: one embedding table per feature (index 0 = unknown/unseen).
        self.cat_embeddings = nn.ModuleList(
            nn.Embedding(card, d_token) for card in cat_cardinalities
        )
        # Kaiming-uniform init, following the rtdl reference.
        nn.init.kaiming_uniform_(self.num_weight, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.num_bias, a=5 ** 0.5)
        for emb in self.cat_embeddings:
            nn.init.kaiming_uniform_(emb.weight, a=5 ** 0.5)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        tokens = []
        if self.n_num:
            # (B, n_num, 1) * (n_num, d) + (n_num, d) -> (B, n_num, d)
            tokens.append(x_num.unsqueeze(-1) * self.num_weight + self.num_bias)
        for j, emb in enumerate(self.cat_embeddings):
            tokens.append(emb(x_cat[:, j]).unsqueeze(1))  # (B, 1, d)
        return torch.cat(tokens, dim=1)  # (B, n_features, d)


class FTTransformer(nn.Module):
    def __init__(self, n_num, cat_cardinalities, cfg):
        super().__init__()
        d = cfg["d_token"]
        self.tokenizer = FeatureTokenizer(n_num, cat_cardinalities, d)
        self.cls = nn.Parameter(torch.empty(1, 1, d))
        nn.init.kaiming_uniform_(self.cls, a=5 ** 0.5)

        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg["n_heads"],
            dim_feedforward=int(d * cfg["ffn_factor"]),
            dropout=cfg["ffn_dropout"],
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm — stabler, matches the FT-Transformer recipe
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=cfg["n_blocks"], enable_nested_tensor=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, 1),
        )

    def forward(self, x_num, x_cat):
        tok = self.tokenizer(x_num, x_cat)                       # (B, F, d)
        cls = self.cls.expand(tok.size(0), -1, -1)               # (B, 1, d)
        h = self.encoder(torch.cat([cls, tok], dim=1))           # (B, 1+F, d)
        return self.head(h[:, 0]).squeeze(-1)                    # read off [CLS]


def _loader(Xnum, Xcat, y, batch_size, shuffle):
    ds = TensorDataset(
        torch.from_numpy(Xnum).float(),
        torch.from_numpy(Xcat).long(),
        torch.from_numpy(y).float(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def _predict_scores(model, Xnum, Xcat, device, batch_size) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(Xnum), batch_size):
        xn = torch.from_numpy(Xnum[i:i + batch_size]).float().to(device)
        xc = torch.from_numpy(Xcat[i:i + batch_size]).long().to(device)
        out.append(torch.sigmoid(model(xn, xc)).cpu().numpy())
    return np.concatenate(out)


def run(b: DataBundle | None = None, cfg=None, log_to_mlflow: bool = True) -> dict:
    """Train the FT-Transformer with early stopping on val ROC-AUC. Returns test metrics."""
    cfg = cfg or load_config()
    b = b or build_bundle(cfg)
    fcfg = cfg["ft_transformer"]

    torch.manual_seed(fcfg["random_state"])
    device = _pick_device(fcfg["device"])
    print(f"[ft] device={device}")

    model = FTTransformer(len(b.num_cols), b.cat_cardinalities, fcfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[ft] parameters: {n_params:,}")

    # Class imbalance -> positive-class weight in the loss.
    pos = float((b.y_train == 1).sum())
    neg = float((b.y_train == 0).sum())
    pos_weight = torch.tensor([neg / pos if pos else 1.0], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=fcfg["lr"],
                            weight_decay=fcfg["weight_decay"])

    train_loader = _loader(b.Xnum_train, b.Xcat_train, b.y_train,
                           fcfg["batch_size"], shuffle=True)

    best_auc, best_state, patience = -1.0, None, 0
    for epoch in range(1, fcfg["max_epochs"] + 1):
        model.train()
        for xn, xc, yy in train_loader:
            xn, xc, yy = xn.to(device), xc.to(device), yy.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xn, xc), yy)
            loss.backward()
            opt.step()

        val_scores = _predict_scores(model, b.Xnum_val, b.Xcat_val, device,
                                     fcfg["batch_size"])
        val_auc = binary_metrics(b.y_val, val_scores)["roc_auc"]
        print(f"[ft] epoch {epoch:02d}  val ROC-AUC={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc, patience = val_auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= fcfg["patience"]:
                print(f"[ft] early stop at epoch {epoch} (best val AUC={best_auc:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_scores = _predict_scores(model, b.Xnum_test, b.Xcat_test, device,
                                  fcfg["batch_size"])
    metrics = binary_metrics(b.y_test, test_scores)
    print(f"[ft] test ROC-AUC={metrics['roc_auc']:.4f} "
          f"PR-AUC={metrics['pr_auc']:.4f} F1={metrics['f1']:.4f}")

    if log_to_mlflow:
        mcfg = cfg["mlflow"]
        mlflow.set_tracking_uri(mcfg["tracking_uri"])
        mlflow.set_experiment(mcfg["experiment"])
        with mlflow.start_run(run_name="ft-transformer"):
            mlflow.log_param("model", "ft_transformer")
            mlflow.log_params({k: fcfg[k] for k in
                               ("d_token", "n_blocks", "n_heads", "lr", "batch_size")})
            mlflow.log_param("n_params", n_params)
            mlflow.log_metric("best_val_roc_auc", best_auc)
            mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
    return metrics


def main() -> int:
    # Runs as its own process (see compare.py) — torch owns the only OpenMP runtime
    # here, so no clash with XGBoost. Deterministic build_bundle => identical split.
    cfg = load_config()
    metrics = run(cfg=cfg)
    out = resolve(cfg["paths"]["reports"]) / "ft_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print(f"[ft] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
