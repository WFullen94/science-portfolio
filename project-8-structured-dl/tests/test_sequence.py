"""Stage-2 tests: sequence data integrity + model shapes.

Uses a small synthetic window set (no network) and cheap CPU forward passes.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from structdl.config import load_config
from structdl.gru_baseline import GRUClassifier
from structdl.sequence_data import build_sequence_bundle
from structdl.ts_transformer import TSTransformer


@pytest.fixture
def cfg():
    c = copy.deepcopy(load_config())
    c["sequence"]["n_windows"] = 800
    return c


def test_bundle_shapes_and_split(cfg):
    b = build_sequence_bundle(cfg)
    T, F = cfg["sequence"]["seq_len"], 5
    total = len(b.ytrain) + len(b.yval) + len(b.ytest)
    assert total == 800
    assert b.Xtrain.shape[1:] == (T, F)
    assert b.Xtest.shape[1:] == (T, F)
    assert len(b.Xtrain) == len(b.ytrain)


def test_standardized_on_train(cfg):
    b = build_sequence_bundle(cfg)
    flat = b.Xtrain.reshape(-1, b.Xtrain.shape[-1])
    assert np.allclose(flat.mean(axis=0), 0, atol=1e-4)
    assert np.allclose(flat.std(axis=0), 1, atol=1e-4)


def test_both_classes_present(cfg):
    b = build_sequence_bundle(cfg)
    for y in (b.ytrain, b.yval, b.ytest):
        assert set(np.unique(y)) == {0, 1}


def test_ts_transformer_forward_shape(cfg):
    b = build_sequence_bundle(cfg)
    model = TSTransformer(len(b.features), cfg["ts_transformer"])
    x = torch.from_numpy(b.Xtrain[:8]).float()
    assert model(x).shape == (8,)  # one logit per window


def test_gru_forward_shape(cfg):
    b = build_sequence_bundle(cfg)
    model = GRUClassifier(len(b.features), cfg["gru"])
    x = torch.from_numpy(b.Xtrain[:8]).float()
    assert model(x).shape == (8,)


def test_distributed_module_imports():
    # Real multi-process DDP is too heavy for CI; just guard against import/API
    # breakage. The DDP path is verified manually via `make ddp` (CPU/gloo).
    import structdl.train_distributed as td
    assert hasattr(td, "_worker") and hasattr(td, "_wrap") and hasattr(td, "_evaluate")
