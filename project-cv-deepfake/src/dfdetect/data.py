"""Stage 1 — Data: download the real/fake face pool + held-out test set, verify, split.

Downloads val.zip (the working pool, re-split into our own train/val below) and
test.zip (held out, untouched, never used for training or model selection) from the
HF-hosted deepfake_face_classification dataset. Both are real images derived from
DF40 (40 distinct deepfake generation techniques) — no synthetic placeholder data.
`prepare()` also carves a tiny bundled sample so CI/tests never need the real download.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import numpy as np
import requests

from dfdetect.config import load_config, resolve

CLASSES = ["real", "fake"]  # label 0=real, 1=fake


def list_items(class_dirs: dict[str, Path], max_per_class: int | None = None) -> list[tuple]:
    """[(path, label)] with label 1=fake, 0=real (matches CLASSES order). Pure —
    no torch — so split/leakage logic is testable without a heavy CI dependency.
    """
    items = []
    for label, cls in enumerate(CLASSES):
        paths = sorted(class_dirs[cls].glob("*"))
        if max_per_class:
            paths = paths[:max_per_class]
        items += [(p, label) for p in paths]
    return items


def stratified_split(items: list[tuple], val_fraction: float, seed: int):
    """Stratified per-class split so train/val get proportional real/fake — pure numpy."""
    rng = np.random.default_rng(seed)
    by_label: dict[int, list] = {0: [], 1: []}
    for p, y in items:
        by_label[y].append((p, y))
    train, val = [], []
    for y, group in by_label.items():
        idx = rng.permutation(len(group))
        n_val = int(round(len(group) * val_fraction))
        val += [group[i] for i in idx[:n_val]]
        train += [group[i] for i in idx[n_val:]]
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def _download(cfg, name: str) -> Path:
    dcfg = cfg["dataset"]
    raw = resolve(dcfg["raw_dir"])
    raw.mkdir(parents=True, exist_ok=True)
    dest = raw / name
    if dest.exists():
        print(f"[data] using cached {dest}")
        return dest
    url = f"{dcfg['base_url']}/{name}"
    print(f"[data] downloading {url} ...")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    print(f"[data] downloaded {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
    return dest


def _extract(zip_path: Path, out_dir: Path) -> Path:
    if out_dir.exists() and any(out_dir.rglob("*.jpg")):
        print(f"[data] already extracted -> {out_dir}")
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[data] extracting {zip_path.name} -> {out_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    return out_dir


def _find_class_dirs(root: Path) -> dict[str, Path]:
    """The zip's internal layout can nest (e.g. val/real/*.jpg or real/*.jpg) —
    search for directories literally named 'real' / 'fake' rather than assuming depth.
    """
    found = {}
    for cls in CLASSES:
        matches = [p for p in root.rglob(cls) if p.is_dir()]
        if not matches:
            raise FileNotFoundError(f"no '{cls}' directory found under {root}")
        # Prefer the one with the most images if multiple match.
        found[cls] = max(matches, key=lambda p: sum(1 for _ in p.glob("*")))
    return found


def prepare(cfg=None) -> dict:
    cfg = cfg or load_config()
    dcfg = cfg["dataset"]

    pool_zip = _download(cfg, dcfg["pool_zip"])
    test_zip = _download(cfg, dcfg["test_zip"])
    pool_dir = _extract(pool_zip, resolve(dcfg["raw_dir"]) / "pool")
    test_dir = _extract(test_zip, resolve(dcfg["raw_dir"]) / "test")

    pool_classes = _find_class_dirs(pool_dir)
    test_classes = _find_class_dirs(test_dir)

    counts = {
        "pool": {c: sum(1 for _ in pool_classes[c].glob("*")) for c in CLASSES},
        "test": {c: sum(1 for _ in test_classes[c].glob("*")) for c in CLASSES},
    }
    print(f"[data] pool: {counts['pool']}  test: {counts['test']}")
    return {"pool": pool_classes, "test": test_classes, "counts": counts}


def build_sample(cfg=None) -> Path:
    """Copy a handful of real+fake images into data/sample for CI/tests."""
    cfg = cfg or load_config()
    dcfg = cfg["dataset"]
    n = dcfg["sample_per_class"]
    paths = prepare(cfg)
    out = resolve("data/sample")
    for cls in CLASSES:
        dst = out / cls
        dst.mkdir(parents=True, exist_ok=True)
        imgs = sorted(paths["test"][cls].glob("*"))[:n]
        for img in imgs:
            shutil.copy(img, dst / img.name)
    print(f"[data] wrote {n}/class sample -> {out}")
    return out


def main() -> int:
    cfg = load_config()
    prepare(cfg)
    build_sample(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
