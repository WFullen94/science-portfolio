# Causal Inference — Does MFA Reduce Account Compromise?

An end-to-end **causal inference** case study: estimate the effect of enabling MFA on account
compromise from *observational* data, where naive analysis is badly misled by confounding.

To have ground truth to check against, the data is **simulated with a known effect** — the honest,
standard way to validate a causal estimator. Account **risk** is a confounder: risky accounts both
get MFA more *and* get compromised more.

## The headline

| | ATE (effect on compromise probability) |
|---|---|
| **True effect** (MFA is protective) | **−0.164** |
| **Naive** mean(Y\|MFA) − mean(Y\|no MFA) | **+0.053** ❗ |

The confounding is strong enough to **flip the sign** — a naive analyst would conclude *MFA increases
compromise* (Simpson's paradox). That's the trap causal methods exist to avoid.

## Recovering the truth

From observed data only (confounders, treatment, outcome):

| Method | ATE | error vs true |
|--------|-----|---------------|
| Regression adjustment (g-computation) | −0.156 | +0.008 |
| Inverse propensity weighting (IPW) | −0.157 | +0.007 |
| AIPW (doubly robust) | −0.157 | +0.007 |
| DoWhy (backdoor linear regression) | −0.161 | +0.003 |

All recover the true ≈ −0.16. **DoWhy refutations pass:** adding a random common cause leaves the
estimate unchanged (robust); a placebo (permuted) treatment drops the effect to ≈ 0.

## Who benefits most? (heterogeneous effects)

A T-learner estimates the per-account effect (CATE). MFA prevents far more compromises for high-risk
accounts — an actionable rollout priority:

| Risk group | estimated effect | true effect |
|------------|------------------|-------------|
| Q1 (low)   | −0.060 | −0.068 |
| Q2         | −0.134 | −0.147 |
| Q3         | −0.202 | −0.215 |
| Q4 (high)  | −0.240 | −0.240 |

Correlation(estimated, true CATE) = **0.98**.

## Run it

```bash
python -m causalmfa.simulate        # generate data; show naive is biased
python -m causalmfa.estimate        # 4 estimators + DoWhy refutation
python -m causalmfa.heterogeneity   # who benefits most (CATE by risk)
```

## Layout

```
conf/config.yaml              DGP coefficients (confounding + true effect), on the logit scale
src/causalmfa/
  simulate.py      simulate with known potential outcomes -> naive bias   [stage 1]
  estimate.py      regression adj / IPW / AIPW / DoWhy + refutation        [stage 2]
  heterogeneity.py T-learner CATE, validated vs true per-unit effect       [stage 3]
```

## The interview framing
> "I framed a security question causally — does MFA reduce compromise — and showed that naive
> analysis is confounded by account risk badly enough to flip the sign. I recovered the true effect
> four ways (regression adjustment, IPW, doubly-robust, DoWhy), validated it with placebo and
> random-common-cause refutations, and used a T-learner to show the effect is strongest for
> high-risk accounts. I validated every estimator against a known ground-truth ATE."
