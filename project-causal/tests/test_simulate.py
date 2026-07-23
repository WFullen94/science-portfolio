"""Simulation tests — the confounding must be real and the truth known."""

from causalmfa.config import load_config
from causalmfa.simulate import simulate


def test_columns_and_binary_fields():
    df, _ = simulate(load_config())
    assert {"risk", "tenure", "prior_incidents", "mfa", "compromised"}.issubset(df.columns)
    assert df["mfa"].isin([0, 1]).all()
    assert df["compromised"].isin([0, 1]).all()


def test_mfa_is_protective_but_naive_is_biased():
    _, truth = simulate(load_config())
    # By construction MFA reduces compromise (negative ATE)...
    assert truth["true_ate"] < 0
    # ...yet naive comparison is badly biased by risk (here it even flips sign).
    assert abs(truth["naive_bias"]) > 0.05
    assert truth["naive_ate"] > truth["true_ate"]
