"""Stage 1 — Simulate the MFA/compromise data with a KNOWN effect + confounding.

We generate both potential outcomes (compromise if-treated Y1 and if-untreated Y0)
for every account, so the true ATE = mean(Y1) - mean(Y0) is known exactly. We then
observe only the outcome consistent with each account's actual MFA status.

Because high-risk accounts both get MFA more AND get compromised more (risk is a
confounder), the naive comparison mean(Y|MFA) - mean(Y|no MFA) is biased — it
understates MFA's protective effect. Later stages recover the truth by adjusting.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from causalmfa.config import load_config, resolve


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def simulate(cfg) -> tuple[pd.DataFrame, dict]:
    s = cfg["simulation"]
    rng = np.random.default_rng(s["seed"])
    n = s["n"]

    # --- confounders ---
    risk = rng.normal(0, 1, n)
    tenure = rng.normal(0, 1, n)
    prior_incidents = rng.poisson(0.5, n).astype(float)

    # --- treatment assignment (confounded by risk / prior incidents) ---
    ta = s["treatment_assignment"]
    logit_t = (ta["intercept"] + ta["risk"] * risk
               + ta["prior_incidents"] * prior_incidents + ta["tenure"] * tenure)
    mfa = rng.binomial(1, _sigmoid(logit_t))

    # --- potential outcomes ---
    o = s["outcome"]
    base = (o["intercept"] + o["risk"] * risk
            + o["prior_incidents"] * prior_incidents + o["tenure"] * tenure)
    p0 = _sigmoid(base)                          # compromise prob if NOT treated
    p1 = _sigmoid(base + o["mfa_effect_logit"])  # compromise prob if treated
    # Shared noise draw so Y0/Y1 differ only through the treatment effect.
    u = rng.uniform(0, 1, n)
    y0 = (u < p0).astype(int)
    y1 = (u < p1).astype(int)

    compromised = np.where(mfa == 1, y1, y0)

    df = pd.DataFrame({
        "risk": risk, "tenure": tenure, "prior_incidents": prior_incidents,
        "mfa": mfa, "compromised": compromised,
    })

    true_ate = float(y1.mean() - y0.mean())
    naive_ate = float(df.loc[df.mfa == 1, "compromised"].mean()
                      - df.loc[df.mfa == 0, "compromised"].mean())
    truth = {
        "true_ate": round(true_ate, 4),
        "naive_ate": round(naive_ate, 4),
        "naive_bias": round(naive_ate - true_ate, 4),
        "treated_frac": round(float(df.mfa.mean()), 4),
        "n": n,
    }
    return df, truth


def main() -> int:
    cfg = load_config()
    df, truth = simulate(cfg)

    data_path = resolve(cfg["paths"]["data"])
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_path, index=False)

    reports = resolve(cfg["paths"]["reports"]); reports.mkdir(parents=True, exist_ok=True)
    (reports / "sim_truth.json").write_text(json.dumps(truth, indent=2))

    print(f"[sim] wrote {len(df):,} accounts ({truth['treated_frac']:.0%} on MFA) -> {data_path}")
    print(f"[sim] TRUE ATE (mean Y1 - Y0)     : {truth['true_ate']:+.4f}  "
          "(negative = MFA reduces compromise)")
    print(f"[sim] NAIVE estimate (Y|MFA - Y|no): {truth['naive_ate']:+.4f}")
    print(f"[sim] naive bias                   : {truth['naive_bias']:+.4f}  "
          "<- confounding by account risk")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
