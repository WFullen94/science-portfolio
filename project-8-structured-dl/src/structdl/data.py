"""Stage 1 — Data: UNSW-NB15 -> one shared, leakage-free split for both models.

The whole point of this project is a fair head-to-head, so the *data* must be
identical for XGBoost and the FT-Transformer. This module owns:

  * loading UNSW-NB15 (public mirrors, with a schema-accurate synthetic fallback
    so the pipeline always runs offline / in CI),
  * one train/val/test split shared by both models,
  * two *views* of that split — each model gets the encoding it wants, but of the
    exact same rows:
      - XGBoost view: a DataFrame with categoricals as pandas `category` dtype
        (native categorical support, no one-hot blowup).
      - FT-Transformer view: standardized numeric matrix + integer-coded
        categorical matrix + per-feature cardinalities.

Encoders (scaler, category vocabularies) are fit on TRAIN ONLY — val/test are
transformed with them, and unseen categories map to a reserved index. That's the
difference between an honest generalization number and a leaked one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from structdl.config import load_config, resolve

# Reserved index 0 in every categorical vocabulary = "unknown / unseen at train time".
UNKNOWN_IDX = 0


# --------------------------------------------------------------------------- #
# Loading                                                                      #
# --------------------------------------------------------------------------- #
def _load_source(src: dict) -> pd.DataFrame | None:
    """Try one configured source; return a merged train(+test) frame or None.

    `url_csv` reads plain CSV URLs (incl. HuggingFace `resolve/` links) with pandas —
    no huggingface_hub dependency. `test_url` is optional; a single file is fine
    because we merge and re-split downstream anyway.
    """
    if src.get("kind") != "url_csv":
        return None
    try:
        frames = [pd.read_csv(src["train_url"])]
        if src.get("test_url"):
            frames.append(pd.read_csv(src["test_url"]))
        return pd.concat(frames, ignore_index=True)
    except Exception as exc:  # network down, repo moved, schema drift — fall through
        print(f"[data] source failed: {exc}")
    return None


def _synthetic(cfg, n: int) -> pd.DataFrame:
    """Schema-accurate synthetic UNSW-NB15 with a learnable signal.

    Not a benchmark — a guarantee the pipeline runs anywhere. The label depends on
    a few numeric features and categorical levels so both models clear AUC 0.5 and
    smoke tests are meaningful. Deterministic given the configured seed.
    """
    rng = np.random.default_rng(cfg["split"]["random_state"])
    protos = np.array(["tcp", "udp", "arp", "ospf", "icmp"])
    services = np.array(["-", "dns", "http", "ftp", "smtp", "ssh"])
    states = np.array(["FIN", "CON", "INT", "REQ", "RST"])

    proto = rng.choice(protos, n, p=[0.55, 0.30, 0.06, 0.05, 0.04])
    service = rng.choice(services, n, p=[0.40, 0.20, 0.20, 0.08, 0.07, 0.05])
    state = rng.choice(states, n, p=[0.45, 0.25, 0.15, 0.10, 0.05])

    dur = rng.exponential(1.0, n)
    sbytes = rng.exponential(500, n)
    dbytes = rng.exponential(500, n)
    rate = rng.exponential(50, n)
    sttl = rng.integers(0, 255, n).astype(float)
    ct_srv_src = rng.integers(1, 60, n).astype(float)

    # A real-ish decision boundary: attacks skew toward certain states/services,
    # short high-rate flows, and specific TTLs. Logit + noise -> Bernoulli label.
    logit = (
        1.4 * (state == "INT")
        + 1.1 * (service == "-")
        + 0.9 * (proto == "udp")
        + 0.8 * (rate > 60)
        + 0.7 * (sttl > 200)
        - 0.6 * (dur > 1.5)
        - 0.9
        + rng.normal(0, 0.5, n)
    )
    label = (1.0 / (1.0 + np.exp(-logit)) > 0.5).astype(int)
    attack_cat = np.where(label == 1,
                          rng.choice(["Generic", "Exploits", "Fuzzers", "DoS"], n),
                          "Normal")

    return pd.DataFrame({
        "id": np.arange(n),
        "dur": dur, "proto": proto, "service": service, "state": state,
        "sbytes": sbytes, "dbytes": dbytes, "rate": rate, "sttl": sttl,
        "ct_srv_src": ct_srv_src, "attack_cat": attack_cat, "label": label,
    })


def load_raw(cfg=None) -> pd.DataFrame:
    """Return the merged, row-capped UNSW-NB15 frame (cached to parquet)."""
    cfg = cfg or load_config()
    dcfg = cfg["dataset"]
    cache = resolve(cfg["paths"]["cache"])
    if cache.exists():
        print(f"[data] using cached {cache}")
        return pd.read_parquet(cache)

    df = None
    for src in dcfg["sources"]:
        df = _load_source(src)
        if df is not None:
            print(f"[data] loaded {len(df)} rows from {src['kind']}")
            break
    if df is None:
        n = dcfg["synthetic_fallback_rows"]
        print(f"[data] all sources unreachable — synthetic fallback ({n} rows)")
        df = _synthetic(cfg, n)

    df.columns = [c.strip().lower() for c in df.columns]
    # Cap after a deterministic shuffle so the sample is class-representative.
    cap = dcfg.get("max_rows")
    if cap and len(df) > cap:
        df = df.sample(n=cap, random_state=cfg["split"]["random_state"])
    df = df.reset_index(drop=True)

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    print(f"[data] cached {len(df)} rows -> {cache}")
    return df


# --------------------------------------------------------------------------- #
# Split + two views                                                            #
# --------------------------------------------------------------------------- #
@dataclass
class DataBundle:
    # Shared column layout
    num_cols: list[str]
    cat_cols: list[str]
    cat_cardinalities: list[int]  # includes the reserved UNKNOWN slot per feature

    # XGBoost view — DataFrames, categoricals as `category` dtype
    X_train_df: pd.DataFrame
    X_val_df: pd.DataFrame
    X_test_df: pd.DataFrame

    # FT-Transformer view — numeric (standardized) + categorical (int codes)
    Xnum_train: np.ndarray
    Xnum_val: np.ndarray
    Xnum_test: np.ndarray
    Xcat_train: np.ndarray
    Xcat_val: np.ndarray
    Xcat_test: np.ndarray

    # Labels (shared)
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def _split_indices(y: np.ndarray, cfg):
    """Stratified train/val/test indices — computed once, shared by both models."""
    from sklearn.model_selection import train_test_split

    scfg = cfg["split"]
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx, test_size=scfg["test_size"], stratify=y,
        random_state=scfg["random_state"],
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=scfg["val_size"], stratify=y[train_idx],
        random_state=scfg["random_state"],
    )
    return train_idx, val_idx, test_idx


def build_bundle(cfg=None) -> DataBundle:
    cfg = cfg or load_config()
    dcfg = cfg["dataset"]
    df = load_raw(cfg)

    target = dcfg["target"]
    drop = set(dcfg.get("drop_columns", [])) | {target, dcfg.get("multiclass", "")}
    cat_cols = [c for c in dcfg["categorical_columns"] if c in df.columns]
    feat_cols = [c for c in df.columns if c not in drop]
    num_cols = [c for c in feat_cols if c not in cat_cols]

    y = df[target].astype(int).to_numpy()
    tr, va, te = _split_indices(y, cfg)

    # --- XGBoost view: categoricals as pandas category dtype -----------------
    Xdf = df[feat_cols].copy()
    for c in cat_cols:
        Xdf[c] = Xdf[c].astype("category")
    X_train_df, X_val_df, X_test_df = Xdf.iloc[tr], Xdf.iloc[va], Xdf.iloc[te]

    # --- FT view: standardize numerics (fit on TRAIN), int-code categoricals -
    num = df[num_cols].astype(np.float32).to_numpy()
    mean = num[tr].mean(axis=0)
    std = num[tr].std(axis=0)
    std[std == 0] = 1.0  # guard constant columns
    num_std = (num - mean) / std

    # Vocab per categorical from TRAIN only; index 0 reserved for unseen.
    cat_codes = np.zeros((len(df), len(cat_cols)), dtype=np.int64)
    cardinalities = []
    for j, c in enumerate(cat_cols):
        vocab = {v: i + 1 for i, v in enumerate(sorted(df[c].iloc[tr].unique()))}
        cat_codes[:, j] = df[c].map(lambda v: vocab.get(v, UNKNOWN_IDX)).to_numpy()
        cardinalities.append(len(vocab) + 1)  # +1 for the reserved unknown slot

    return DataBundle(
        num_cols=num_cols, cat_cols=cat_cols, cat_cardinalities=cardinalities,
        X_train_df=X_train_df, X_val_df=X_val_df, X_test_df=X_test_df,
        Xnum_train=num_std[tr], Xnum_val=num_std[va], Xnum_test=num_std[te],
        Xcat_train=cat_codes[tr], Xcat_val=cat_codes[va], Xcat_test=cat_codes[te],
        y_train=y[tr], y_val=y[va], y_test=y[te],
    )


def main() -> int:
    cfg = load_config()
    b = build_bundle(cfg)
    print(f"[data] features: {len(b.num_cols)} numeric + {len(b.cat_cols)} categorical")
    print(f"[data] cat cardinalities: {dict(zip(b.cat_cols, b.cat_cardinalities))}")
    print(f"[data] split: train={len(b.y_train)} val={len(b.y_val)} test={len(b.y_test)}")
    print(f"[data] train attack rate: {b.y_train.mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
