# Structured-Data DL — Transformers vs the right baseline

Two **controlled head-to-heads** on structured NIDS data, each pitting a Transformer against the
strong classical baseline for that data shape — and reporting the honest result:

- **Stage 1 (tabular)** — hand-rolled **FT-Transformer** vs a tuned **XGBoost**. *The tree wins,
  narrowly* (0.990 vs 0.986 ROC-AUC) — as the tabular literature predicts.
- **Stage 2 (sequence)** — a **time-series Transformer** vs a **GRU**. *Attention wins decisively on
  long windows* (0.996 vs 0.607 at length 256) — recurrence forgets, attention doesn't.

The through-line isn't "deep learning wins" — it's **match the inductive bias to the data, and prove
it with a fair comparison** (identical splits, leakage-free encoders, comparable tuning). Knowing
*when a transformer is and isn't worth it* is the actual decision an applied scientist owns.

---

# Stage 1 — Tabular: FT-Transformer vs XGBoost on UNSW-NB15

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
conf/config.yaml            single source of truth (data, splits, all models, MLflow)
src/structdl/
  metrics.py                shared binary-classification metrics (both stages)
  # stage 1 — tabular
  data.py                   UNSW-NB15 -> one shared leakage-free split, two model views   [stage 1]
  xgb_baseline.py           XGBoost + Optuna CV search (the baseline)                      [stage 1a]
  ft_transformer.py         hand-rolled FT-Transformer (feature tokenizer + CLS)           [stage 1b]
  compare.py                one split -> both models (subprocess-isolated) -> report       [stage 1c]
  # stage 2 — sequence
  sequence_data.py          windowed telemetry w/ temporal-burst signal, shared split      [stage 2]
  seq_train.py              shared train/eval/MLflow loop for both sequence models
  gru_baseline.py           recurrent GRU classifier (the baseline)                        [stage 2a]
  ts_transformer.py         time-series Transformer (proj + pos-enc + CLS)                 [stage 2b]
  ts_compare.py             single-length GRU vs TS-Transformer head-to-head               [stage 2c]
  ts_sweep.py               length sweep — attention vs recurrence (headline)              [stage 2d]
tests/                      data integrity (leakage, shapes) + model shapes, both stages
```

## The interview framing

> "I ran two controlled head-to-heads on structured NIDS data, each transformer against the right
> classical baseline. On tabular data a tuned XGBoost edged my FT-Transformer (0.990 vs 0.986) —
> trees still win there. On sequential telemetry the story flipped: a time-series Transformer held
> ~1.0 ROC-AUC while a GRU collapsed to near-random once windows got long enough that its last-hidden
> read-out forgot the burst. Same split, leakage-free encoders, comparable tuning both times. The
> point isn't that deep learning wins — it's matching the inductive bias to the data and *proving*
> which model to ship, which is the decision an applied scientist actually owns."

---

# Stage 2 — Sequence detection: time-series Transformer vs GRU

Stage 1 asked *"transformer or tree on a table?"* Stage 2 asks a different question on
**sequential** data: *"attention or recurrence on a window of telemetry?"* — and the answer
flips, which is the point of running both.

## The task (designed so order matters)

Windows of per-step telemetry where an attack is a **short, absolutely-sized burst** (a 2–4 step
spike) placed anywhere in the window; benign windows are smooth, sometimes with broad *evenly-spread*
elevation (so a mean-pooling model can't cheat on totals). Detecting it requires reading the
temporal *shape*, not the average. Synthetic + deterministic so it always runs
([sequence_data.py](src/structdl/sequence_data.py)); leakage-free standardization fit on train only.

## The headline — where attention beats recurrence (length sweep)

Because the burst is a *fixed-size* event, a longer window makes it a sparser needle sitting further
from the sequence end. A GRU classifies from its **last hidden state**, so an early burst in a long
window gets forgotten; the Transformer **attends** to every step, so position is irrelevant. Run the
head-to-head across window lengths ([ts_sweep.py](src/structdl/ts_sweep.py)):

| seq_len | GRU ROC-AUC | TS-Transformer ROC-AUC | Δ (TS − GRU) |
|---|---|---|---|
| 32 | 1.0000 | 1.0000 | −0.0000 |
| 64 | 1.0000 | 1.0000 | −0.0000 |
| 128 | 1.0000 | 0.9999 | −0.0001 |
| **256** | **0.6071** | **0.9964** | **+0.3892** |

**The GRU ties the Transformer up to length 128, then collapses to near-random at 256** while the
Transformer holds. That's the vanishing-memory failure of a recurrent read-out made visible — and
the mirror image of stage 1: *the tree beat the transformer on a table; attention crushes recurrence
on long sequences.* Same lesson both times — **match the inductive bias to the data**, and prove it
rather than assume it.

## Architecture

Both are pure-torch binary sequence classifiers sharing one training loop
([seq_train.py](src/structdl/seq_train.py)) so the comparison differs only in the model:

- **TS-Transformer** ([ts_transformer.py](src/structdl/ts_transformer.py)) — per-step linear
  projection → sinusoidal positional encoding → learned `[CLS]` → Transformer encoder → read off `[CLS]`.
- **GRU** ([gru_baseline.py](src/structdl/gru_baseline.py)) — multi-layer GRU, classify from the
  last-step hidden state (the standard, and the thing that fails at length).

```bash
make ts-sweep     # the length sweep (headline) -> data/reports/ts_sweep.json
make ts-compare   # single-length GRU vs TS-Transformer head-to-head
```

---

## Roadmap (this project)

- ✅ **Stage 1** — tabular: FT-Transformer vs XGBoost.
- ✅ **Stage 2** — sequence: time-series Transformer vs GRU (length sweep).
- **Stage 3 — distributed training**: take a training loop from single-device to **DDP → FSDP**,
  documenting the sharding story.
