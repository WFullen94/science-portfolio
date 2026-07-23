"""The T-learner should recover the heterogeneous effect the simulator built in."""

import numpy as np

from causalmfa.config import load_config
from causalmfa.heterogeneity import t_learner_cate
from causalmfa.simulate import potential_outcomes_frame


def test_t_learner_tracks_true_cate():
    cfg = load_config()
    full = potential_outcomes_frame(cfg)
    cf, t, y = cfg["confounders"], cfg["treatment"], cfg["outcome"]
    est = t_learner_cate(full[cf + [t, y]].copy(), t, y, cf)
    true = (full["p1"] - full["p0"]).to_numpy()
    assert np.corrcoef(est, true)[0, 1] > 0.9
