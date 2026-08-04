"""Stage 2b — the baseline: a GRU sequence classifier.

The recurrent classic, and the bar the time-series Transformer has to clear. Reads
the window step by step; the final hidden state feeds a linear head. Same question
as stage 1 in a new guise: does the Transformer actually beat the simpler,
well-understood baseline (there a tree, here a GRU)?
"""

from __future__ import annotations

import json

import torch
import torch.nn as nn

from structdl.config import load_config, resolve
from structdl.sequence_data import build_sequence_bundle
from structdl.seq_train import train_sequence_model


class GRUClassifier(nn.Module):
    def __init__(self, n_features: int, cfg):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["n_layers"],
            batch_first=True,
            dropout=cfg["dropout"] if cfg["n_layers"] > 1 else 0.0,
            bidirectional=cfg["bidirectional"],
        )
        out_dim = cfg["hidden_size"] * (2 if cfg["bidirectional"] else 1)
        self.head = nn.Sequential(nn.LayerNorm(out_dim), nn.Linear(out_dim, 1))

    def forward(self, x):                       # x: (B, T, F)
        out, h = self.gru(x)                    # out: (B, T, H*dir)
        last = out[:, -1]                       # last-step representation
        return self.head(last).squeeze(-1)


def run(b=None, cfg=None, log_to_mlflow: bool = True) -> dict:
    cfg = cfg or load_config()
    b = b or build_sequence_bundle(cfg)
    model = GRUClassifier(len(b.features), cfg["gru"])
    return train_sequence_model(
        model, b, cfg["gru"], run_name="gru-baseline",
        experiment=cfg["mlflow"]["experiment"], tracking_uri=cfg["mlflow"]["tracking_uri"],
        log_to_mlflow=log_to_mlflow,
    )


def main() -> int:
    cfg = load_config()
    metrics = run(cfg=cfg)
    out = resolve(cfg["paths"]["reports"]) / "gru_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print(f"[gru-baseline] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
