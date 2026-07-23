"""Stage 3 — Heterogeneous treatment effects: who benefits most from MFA?

A single average (ATE) can hide that MFA helps some accounts far more than others.
We estimate the per-account effect (CATE) with a T-learner — one outcome model on
treated accounts, one on controls, and take the predicted difference for everyone —
then validate it against the true per-account effect the simulator knows.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from causalmfa.config import load_config, resolve
from causalmfa.simulate import potential_outcomes_frame


def t_learner_cate(df, t, y, confounders) -> np.ndarray:
    treated, control = df[df[t] == 1], df[df[t] == 0]
    m1 = LogisticRegression(max_iter=1000).fit(treated[confounders], treated[y])
    m0 = LogisticRegression(max_iter=1000).fit(control[confounders], control[y])
    return (m1.predict_proba(df[confounders])[:, 1]
            - m0.predict_proba(df[confounders])[:, 1])


def main() -> int:
    cfg = load_config()
    t, y, cf = cfg["treatment"], cfg["outcome"], cfg["confounders"]

    full = potential_outcomes_frame(cfg)
    df = full[cf + [t, y]].copy()

    est_tau = t_learner_cate(df, t, y, cf)
    true_tau = (full["p1"] - full["p0"]).to_numpy()   # true individual effect
    corr = float(np.corrcoef(est_tau, true_tau)[0, 1])

    quart = pd.qcut(full["risk"], 4, labels=["Q1 (low risk)", "Q2", "Q3", "Q4 (high risk)"])
    table = (pd.DataFrame({"risk_quartile": quart, "estimated": est_tau, "true": true_tau})
             .groupby("risk_quartile", observed=True).mean().round(4))

    print("[cate] per-account effect of MFA on compromise probability, by risk group:")
    print(table.to_string())
    print(f"\n[cate] corr(estimated CATE, true CATE) = {corr:.3f}")
    print("[cate] (more negative = MFA prevents more compromises for that group)")

    reports = resolve(cfg["paths"]["reports"]); reports.mkdir(parents=True, exist_ok=True)
    (reports / "cate_by_risk.json").write_text(json.dumps({
        "corr_est_true": round(corr, 4),
        "by_quartile": {str(k): {"estimated": float(v["estimated"]), "true": float(v["true"])}
                        for k, v in table.iterrows()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
