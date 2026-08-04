"""Stage 2a — the challenger: a time-series Transformer (encoder over the sequence).

Projects each telemetry step to d_model, adds sinusoidal positional encoding (so the
model knows step order — the burst's *position* and *contiguity* are the signal),
prepends a learned [CLS] token, runs a Transformer encoder, and reads the label off
[CLS]. This is the sequence analogue of stage 1's FT-Transformer.
"""

from __future__ import annotations

import json
import math

import torch
import torch.nn as nn

from structdl.config import load_config, resolve
from structdl.sequence_data import build_sequence_bundle
from structdl.seq_train import train_sequence_model


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding added to the projected steps."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TSTransformer(nn.Module):
    def __init__(self, n_features: int, cfg):
        super().__init__()
        d = cfg["d_model"]
        self.input_proj = nn.Linear(n_features, d)
        self.pos = PositionalEncoding(d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg["n_heads"],
            dim_feedforward=int(d * cfg["ffn_factor"]),
            dropout=cfg["dropout"], activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg["n_blocks"],
                                             enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, x):                                  # x: (B, T, F)
        h = self.pos(self.input_proj(x))                   # (B, T, d)
        cls = self.cls.expand(h.size(0), -1, -1)           # (B, 1, d)
        h = self.encoder(torch.cat([cls, h], dim=1))       # (B, 1+T, d)
        return self.head(h[:, 0]).squeeze(-1)              # read off [CLS]


def run(b=None, cfg=None, log_to_mlflow: bool = True) -> dict:
    cfg = cfg or load_config()
    b = b or build_sequence_bundle(cfg)
    model = TSTransformer(len(b.features), cfg["ts_transformer"])
    return train_sequence_model(
        model, b, cfg["ts_transformer"], run_name="ts-transformer",
        experiment=cfg["mlflow"]["experiment"], tracking_uri=cfg["mlflow"]["tracking_uri"],
        log_to_mlflow=log_to_mlflow,
    )


def main() -> int:
    cfg = load_config()
    metrics = run(cfg=cfg)
    out = resolve(cfg["paths"]["reports"]) / "ts_transformer_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print(f"[ts-transformer] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
