"""Stage 2 — Sequence data: windowed telemetry with a *temporal* attack signal.

Stage 1 was tabular (one row = one flow). Stage 2 asks a different question: can a
Transformer read a *sequence* of telemetry steps and spot an attack from its temporal
shape? To make that a fair test, the label has to depend on order, not just averages.

So the synthetic generator builds fixed-length windows of per-step telemetry where:
  * benign windows are smooth, and may even carry high *total* energy spread evenly;
  * attack windows concentrate that energy into a short contiguous **burst** (a spike
    in rate/bytes over a few consecutive steps).

A bag-of-features model that mean-pools the window sees similar totals for both and
struggles; a sequence model (GRU / Transformer) can see the burst *shape*. That's the
point of the stage. Deterministic given the seed; always runnable (no download), matching
stage 1's philosophy. A real timestamped-flow source can be slotted in later.

Encoders (per-feature standardization) are fit on TRAIN only — val/test transform with
them. Same leakage discipline as stage 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from structdl.config import load_config

FEATURES = ["rate", "sbytes", "dpkts", "dur", "sttl"]


@dataclass
class SeqBundle:
    features: list[str]
    seq_len: int
    Xtrain: np.ndarray  # (N, T, F) float32
    Xval: np.ndarray
    Xtest: np.ndarray
    ytrain: np.ndarray  # (N,)
    yval: np.ndarray
    ytest: np.ndarray


def _synthesize(cfg) -> tuple[np.ndarray, np.ndarray]:
    """Return (X: (N,T,F), y: (N,)) with a temporal-burst attack signal."""
    scfg = cfg["sequence"]
    rng = np.random.default_rng(scfg["random_state"])
    n, T, F = scfg["n_windows"], scfg["seq_len"], len(FEATURES)
    attack_rate = scfg.get("attack_rate", 0.4)

    y = (rng.random(n) < attack_rate).astype(int)
    X = np.zeros((n, T, F), dtype=np.float32)

    for i in range(n):
        # Baseline smooth telemetry (correlated across features via a shared level).
        level = rng.normal(0, 1, F)
        walk = np.cumsum(rng.normal(0, 0.15, (T, F)), axis=0)
        base = level + walk  # (T, F) slow-varying benign signal

        if y[i] == 0:
            # Benign: optionally inject broad, *evenly spread* elevation (high total
            # energy, no burst) so mean-pooling can't separate it from an attack.
            if rng.random() < 0.5:
                base += rng.uniform(0.5, 1.2)
            X[i] = base
        else:
            # Attack: a short, ABSOLUTELY-sized contiguous burst (2–4 steps) placed
            # anywhere in the window. Because burst length does NOT grow with T, a
            # longer window makes the burst a sparser needle — and, crucially, often
            # far from the sequence end. A last-hidden-state GRU must carry that
            # signal across many steps (it forgets); attention reads it directly.
            burst_len = int(rng.integers(2, 5))
            start = rng.integers(0, T - burst_len)
            amp = rng.uniform(scfg.get("burst_amp_low", 2.0),
                              scfg.get("burst_amp_high", 3.0))
            burst = np.zeros((T, F), dtype=np.float32)
            burst[start:start + burst_len, :3] = amp  # rate, sbytes, dpkts spike
            burst[start:start + burst_len, 4] = amp * 0.5  # sttl shift
            X[i] = base + burst

        X[i] += rng.normal(0, scfg.get("noise", 0.2), (T, F))  # observation noise
    return X, y


def _split(y, cfg):
    from sklearn.model_selection import train_test_split

    scfg = cfg["sequence"]
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=scfg["test_size"], stratify=y,
                              random_state=scfg["random_state"])
    tr, va = train_test_split(tr, test_size=scfg["val_size"], stratify=y[tr],
                              random_state=scfg["random_state"])
    return tr, va, te


def build_sequence_bundle(cfg=None) -> SeqBundle:
    cfg = cfg or load_config()
    X, y = _synthesize(cfg)
    tr, va, te = _split(y, cfg)

    # Standardize per feature on TRAIN across (windows × timesteps); apply to all.
    flat = X[tr].reshape(-1, X.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std == 0] = 1.0
    Xn = (X - mean) / std

    return SeqBundle(
        features=FEATURES, seq_len=X.shape[1],
        Xtrain=Xn[tr], Xval=Xn[va], Xtest=Xn[te],
        ytrain=y[tr], yval=y[va], ytest=y[te],
    )


def main() -> int:
    cfg = load_config()
    b = build_sequence_bundle(cfg)
    print(f"[seq] windows: train={len(b.ytrain)} val={len(b.yval)} test={len(b.ytest)}")
    print(f"[seq] shape per split: (N, T={b.seq_len}, F={len(b.features)}) "
          f"features={b.features}")
    print(f"[seq] train attack rate: {b.ytrain.mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
