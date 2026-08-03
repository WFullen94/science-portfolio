"""Data-layer tests: shared split, leakage-free encoders, correct shapes.

Forced onto the synthetic fallback (sources=[]) so the suite never touches the
network — same philosophy as P1/P3's model-free tests.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from structdl.config import load_config
from structdl.data import UNKNOWN_IDX, build_bundle


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(load_config())
    c["dataset"]["sources"] = []                       # force synthetic
    c["dataset"]["synthetic_fallback_rows"] = 3000
    c["dataset"]["max_rows"] = None
    c["paths"]["cache"] = str(tmp_path / "synth.parquet")
    return c


def test_split_sizes_partition_the_data(cfg):
    b = build_bundle(cfg)
    total = len(b.y_train) + len(b.y_val) + len(b.y_test)
    assert total == 3000
    # test is 20% of all; val is 20% of the remaining 80%.
    assert len(b.y_test) == pytest.approx(0.20 * 3000, abs=2)
    assert len(b.y_val) == pytest.approx(0.20 * 0.80 * 3000, abs=2)


def test_views_share_the_same_rows(cfg):
    b = build_bundle(cfg)
    # XGBoost DataFrame view and FT array view must describe the same split.
    assert len(b.X_train_df) == len(b.Xnum_train) == len(b.Xcat_train) == len(b.y_train)
    assert len(b.X_test_df) == len(b.Xnum_test) == len(b.y_test)


def test_numeric_standardized_on_train(cfg):
    b = build_bundle(cfg)
    # Scaler fit on train -> train columns ~0 mean, ~1 std.
    assert np.allclose(b.Xnum_train.mean(axis=0), 0, atol=1e-4)
    assert np.allclose(b.Xnum_train.std(axis=0), 1, atol=1e-4)


def test_no_category_leakage(cfg):
    b = build_bundle(cfg)
    # Vocab is built from train, so every train code is a real (>=1) index —
    # the reserved UNKNOWN slot only ever appears for values unseen at train time.
    assert (b.Xcat_train >= 1).all()
    for j, card in enumerate(b.cat_cardinalities):
        assert b.Xcat_train[:, j].max() < card
        assert b.Xcat_val[:, j].min() >= UNKNOWN_IDX
        assert b.Xcat_test[:, j].max() < card


def test_cardinalities_reserve_unknown_slot(cfg):
    b = build_bundle(cfg)
    # cardinality = (#distinct train values) + 1 reserved unknown slot.
    assert len(b.cat_cardinalities) == len(b.cat_cols)
    assert all(card >= 2 for card in b.cat_cardinalities)
