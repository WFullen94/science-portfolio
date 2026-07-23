"""The causal estimators must recover the truth; the naive one must not.
(Fast manual estimators only — DoWhy's refutation is exercised in the CLI run.)"""

from causalmfa.config import load_config
from causalmfa.estimate import aipw, ipw, naive_ate, regression_adjustment
from causalmfa.simulate import simulate


def test_causal_estimators_recover_truth_naive_does_not():
    cfg = load_config()
    df, truth = simulate(cfg)
    t, y, cf = cfg["treatment"], cfg["outcome"], cfg["confounders"]
    true = truth["true_ate"]

    # Naive is badly biased (here even the wrong sign).
    assert abs(naive_ate(df, t, y) - true) > 0.15

    # Adjusting for confounders recovers the true ATE within a small tolerance.
    for estimator in (regression_adjustment, ipw, aipw):
        est = estimator(df, t, y, cf)
        assert abs(est - true) < 0.03, (estimator.__name__, est, true)
