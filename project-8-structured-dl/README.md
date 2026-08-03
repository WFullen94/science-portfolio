# Structured-Data DL — Tabular Transformer vs XGBoost on UNSW-NB15

A **controlled head-to-head**: a hand-rolled **FT-Transformer** against a tuned **XGBoost**
baseline, on the same intrusion-detection rows, same split, same metrics. The point isn't to
crown deep learning — it's to answer *how close a Transformer gets on tabular data, and where*,
honestly.

## The framing (why this is the interesting question)

The settled result in the tabular literature ([Grinsztajn et al., 2022](https://arxiv.org/abs/2207.08815))
is that **gradient-boosted trees still usually beat deep nets on tabular data.** So a project that
declares "transformer wins!" is either cherry-picked or naive. The defensible version is a *fair*
comparison that reports the honest gap and the conditions under which it narrows — which is what an
applied scientist actually has to decide when someone asks "should we put a neural net on this table?"

**Dataset choice is part of the argument.** P1 (The Spine) used the NetFlow-v2 schema, which is
almost entirely numeric — XGBoost's home turf, where a transformer has nothing to attend over. This
project deliberately uses the **original UNSW-NB15** (45-column academic partition), which keeps real
**categorical** features (`proto`, `service`, `state`). Per-feature attention needs categorical
structure to show what it can do, so this is the dataset that gives the challenger a fair shot.

## The head-to-head (held-out test set)

UNSW-NB15, 60k sampled rows → identical split (38,400 train / 12,000 test), 39 numeric + 3
categorical features. Both runs tracked in **MLflow** under one experiment; numbers land in
[data/reports/headtohead.json](data/reports/headtohead.json).

| model | ROC-AUC | PR-AUC | accuracy | F1 |
|---|---|---|---|---|
| **XGBoost** (Optuna-tuned) | **0.9898** | **0.9944** | **0.9388** | **0.9511** |
| **FT-Transformer** (hand-rolled) | 0.9862 | 0.9924 | 0.9275 | 0.9417 |
| **Δ (XGB − FT)** | +0.0036 | +0.0020 | +0.0113 | +0.0094 |

**The tree wins — narrowly.** XGBoost is ahead on every metric, but by a hair: **+0.0036 ROC-AUC
(~0.4%)**. That is exactly the result the tabular literature predicts — gradient-boosted trees still
edge out deep nets on tabular data — and the honest, defensible headline: *a Transformer gets within
half a percent of a tuned XGBoost on categorical NIDS data, but doesn't beat it, and costs far more
to train (40 epochs on GPU/MPS vs a CPU Optuna search).* On this data, XGBoost is the right call —
and knowing **that** is the point of the exercise.

## FT-Transformer in one paragraph

Every feature becomes a `d_token` vector: a **numeric** feature `x` maps to `x·W + b` (a learned
per-feature affine), a **categorical** value maps to an `Embedding[value]` lookup. Stack those tokens,
prepend a learned **`[CLS]`** token, run a Transformer encoder so features attend to each other, and
read the prediction off the final `[CLS]`. That feature-to-feature attention is the thing a tree
can't do. I hand-roll the **feature tokenizer** and CLS pooling (the tabular-specific parts) in
[ft_transformer.py](src/structdl/ft_transformer.py); the attention/FFN backbone is torch's own
pre-norm encoder layer.

## What makes it a fair fight

One split, built once, handed to both models — the crux of an honest comparison
([data.py](src/structdl/data.py)):

- **Identical rows** for both models; each gets the *encoding* it wants (XGBoost: pandas `category`
  dtype + native `enable_categorical`; FT-Transformer: standardized numerics + integer-coded
  categoricals) but of the *same* split.
- **Leakage-free encoders** — the scaler and category vocabularies are fit on **train only**; unseen
  categories at val/test map to a reserved "unknown" index.
- **Same metrics, same threshold** via a shared [metrics.py](src/structdl/metrics.py) (ROC-AUC and
  PR-AUC are threshold-free; accuracy/precision/recall/F1 at 0.5).
- **Comparable tuning effort** — XGBoost gets an Optuna CV search (the P1 pattern); the FT-Transformer
  early-stops on validation ROC-AUC.

## Run it

```bash
make setup      # venv + deps
make compare    # build shared split -> XGBoost + FT-Transformer -> head-to-head table
# or piecewise:
make data       # inspect the prepared split
make xgb        # baseline only
make ft         # transformer only
make test       # data-integrity + model-shape tests (no network, synthetic fallback)
```

The dataset auto-downloads from public UNSW-NB15 mirrors; if none is reachable (offline / CI), the
pipeline falls back to a **schema-accurate synthetic sample** so it always runs.

## Layout

```
conf/config.yaml            single source of truth (dataset, shared split, both models, MLflow)
src/structdl/
  data.py                   UNSW-NB15 -> one shared leakage-free split, two model views   [stage 1]
  metrics.py                shared binary-classification metrics
  xgb_baseline.py           XGBoost + Optuna CV search (the baseline)                      [stage 1a]
  ft_transformer.py         hand-rolled FT-Transformer (feature tokenizer + CLS)           [stage 1b]
  compare.py                one split -> both models -> head-to-head + report              [stage 1c]
tests/                      data integrity (leakage, shapes) + model shapes
```

## The interview framing

> "I ran a controlled FT-Transformer vs XGBoost head-to-head on categorical NIDS data — same split,
> leakage-free encoders fit on train only, comparable tuning budgets. I picked the categorical
> UNSW-NB15 partition on purpose, because a tabular transformer's per-feature attention has nothing
> to do on the pure-numeric NetFlow schema. I report the honest gap rather than a rigged win —
> which is the decision an applied scientist actually owns: *is a neural net worth it on this table?*"

## Roadmap (this project, later stages)

- **Stage 2 — time-series transformer** on sequential telemetry (windowed flows).
- **Stage 3 — distributed training**: take this FT-Transformer training loop from single-device to
  **DDP → FSDP**, documenting the sharding story.
