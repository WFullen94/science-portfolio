"""Stage 2 — Estimate the causal effect and recover the truth.

From the OBSERVED data (confounders, treatment, outcome) — never the potential
outcomes — estimate the ATE four ways, then validate with DoWhy's refutation tests:

  naive                 : mean(Y|T=1) - mean(Y|T=0)          (biased baseline)
  regression adjustment : outcome model, predict under T=1/0 (g-computation)
  IPW                   : inverse propensity weighting
  AIPW (doubly robust)  : combines outcome + propensity models
  DoWhy                 : model -> identify (backdoor) -> estimate -> refute

The proper estimators should land near the true ATE; the naive one should not.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from causalmfa.config import load_config, resolve

logging.getLogger("dowhy").setLevel(logging.ERROR)


def naive_ate(df, t, y):
    return float(df.loc[df[t] == 1, y].mean() - df.loc[df[t] == 0, y].mean())


def _propensity(df, t, confounders):
    e = LogisticRegression(max_iter=1000).fit(df[confounders], df[t]).predict_proba(df[confounders])[:, 1]
    return np.clip(e, 0.02, 0.98)  # trim to avoid extreme weights


def _outcome_preds(df, t, y, confounders):
    """Fit Y ~ T + confounders, predict each unit under T=1 and T=0."""
    X = df[[t] + confounders].copy()
    model = LogisticRegression(max_iter=1000).fit(X, df[y])
    X1, X0 = X.copy(), X.copy()
    X1[t] = 1
    X0[t] = 0
    mu1 = model.predict_proba(X1)[:, 1]
    mu0 = model.predict_proba(X0)[:, 1]
    return mu1, mu0


def regression_adjustment(df, t, y, confounders):
    mu1, mu0 = _outcome_preds(df, t, y, confounders)
    return float(np.mean(mu1 - mu0))


def ipw(df, t, y, confounders):
    e = _propensity(df, t, confounders)
    T, Y = df[t].to_numpy(), df[y].to_numpy()
    # Hajek (normalized) IPW.
    w1, w0 = T / e, (1 - T) / (1 - e)
    return float((w1 * Y).sum() / w1.sum() - (w0 * Y).sum() / w0.sum())


def aipw(df, t, y, confounders):
    e = _propensity(df, t, confounders)
    mu1, mu0 = _outcome_preds(df, t, y, confounders)
    T, Y = df[t].to_numpy(), df[y].to_numpy()
    dr = (mu1 - mu0) + T * (Y - mu1) / e - (1 - T) * (Y - mu0) / (1 - e)
    return float(np.mean(dr))


def dowhy_estimate(df, t, y, confounders):
    from dowhy import CausalModel

    model = CausalModel(data=df, treatment=t, outcome=y, common_causes=confounders)
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    # Linear regression on the backdoor set: coefficient on T is the ATE (a linear
    # probability model here). Refuters behave cleanly with it, unlike PSW.
    estimate = model.estimate_effect(
        identified, method_name="backdoor.linear_regression", target_units="ate"
    )
    ate = float(estimate.value)

    # Refutation: robustness checks that a valid causal estimate should pass.
    rnd = model.refute_estimate(identified, estimate, method_name="random_common_cause")
    plc = model.refute_estimate(identified, estimate,
                                method_name="placebo_treatment_refuter",
                                placebo_type="permute", num_simulations=20)
    refutations = {
        "random_common_cause_new_effect": round(float(rnd.new_effect), 4),
        "placebo_new_effect": round(float(plc.new_effect), 4),
    }
    return ate, refutations


def main() -> int:
    cfg = load_config()
    t, y, confounders = cfg["treatment"], cfg["outcome"], cfg["confounders"]
    df = pd.read_csv(resolve(cfg["paths"]["data"]))
    truth = json.loads((resolve(cfg["paths"]["reports"]) / "sim_truth.json").read_text())
    true_ate = truth["true_ate"]

    results = {
        "true_ate": true_ate,
        "naive": round(naive_ate(df, t, y), 4),
        "regression_adjustment": round(regression_adjustment(df, t, y, confounders), 4),
        "ipw": round(ipw(df, t, y, confounders), 4),
        "aipw_doubly_robust": round(aipw(df, t, y, confounders), 4),
    }
    dowhy_ate, refutations = dowhy_estimate(df, t, y, confounders)
    results["dowhy_linear"] = round(dowhy_ate, 4)
    results["refutations"] = refutations

    print(f"[est] TRUE ATE                 : {true_ate:+.4f}")
    print(f"[est] naive (biased)           : {results['naive']:+.4f}")
    for name in ("regression_adjustment", "ipw", "aipw_doubly_robust", "dowhy_linear"):
        err = results[name] - true_ate
        print(f"[est] {name:22s}: {results[name]:+.4f}  (err {err:+.4f})")
    print(f"[est] refute random cause      : {refutations['random_common_cause_new_effect']:+.4f}"
          "  (should stay near the estimate -> robust)")
    print(f"[est] refute placebo treatment : {refutations['placebo_new_effect']:+.4f}"
          "  (should be ~0 -> passes)")

    out = resolve(cfg["paths"]["reports"]) / "causal_estimates.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"[est] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
