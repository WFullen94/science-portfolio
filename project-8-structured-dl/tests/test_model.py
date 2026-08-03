"""Model-shape tests: the tokenizer and full FT-Transformer produce correct shapes.

Cheap forward passes on random tensors — no training, CI-friendly.
"""

from __future__ import annotations

import numpy as np
import torch

from structdl.ft_transformer import FeatureTokenizer, FTTransformer
from structdl.metrics import binary_metrics

FCFG = {
    "d_token": 32, "n_blocks": 2, "n_heads": 4, "ffn_factor": 2.0,
    "attention_dropout": 0.1, "ffn_dropout": 0.1, "residual_dropout": 0.0,
}


def test_tokenizer_emits_one_token_per_feature():
    tok = FeatureTokenizer(n_num=5, cat_cardinalities=[4, 7], d_token=32)
    x_num = torch.randn(8, 5)
    x_cat = torch.randint(0, 4, (8, 2))
    out = tok(x_num, x_cat)
    assert out.shape == (8, 5 + 2, 32)  # (batch, n_features, d_token)


def test_ft_transformer_forward_is_per_row_logit():
    model = FTTransformer(n_num=5, cat_cardinalities=[4, 7], cfg=FCFG)
    x_num = torch.randn(8, 5)
    x_cat = torch.randint(0, 4, (8, 2))
    out = model(x_num, x_cat)
    assert out.shape == (8,)  # one logit per row


def test_handles_all_numeric_and_all_categorical():
    # No categoricals.
    m1 = FTTransformer(n_num=6, cat_cardinalities=[], cfg=FCFG)
    assert m1(torch.randn(4, 6), torch.zeros(4, 0, dtype=torch.long)).shape == (4,)
    # No numerics.
    m2 = FTTransformer(n_num=0, cat_cardinalities=[3, 3], cfg=FCFG)
    assert m2(torch.zeros(4, 0), torch.randint(0, 3, (4, 2))).shape == (4,)


def test_metrics_perfect_separation():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    m = binary_metrics(y, scores)
    assert m["roc_auc"] == 1.0
    assert m["f1"] == 1.0
