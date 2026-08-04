"""Stage 1 — Data: verify the VisDrone YOLO dataset and carve a tiny bundled sample.

The full VisDrone-DET dataset (6,471 train / 548 val, 10 classes) already exists in
YOLO format in the computer-vision repo, with images symlinked into the raw 1.6GB
tree — so we *reference* it (config `dataset.yaml`) rather than copy it. This module:

  * `verify()` — confirm the external dataset.yaml resolves and images are reachable;
  * `build_sample()` — copy a handful of val images+labels into data/sample and write a
    self-contained sample dataset.yaml, so CI / smoke tests never need the 1.6GB tree.

Reproduce on another machine: VisDrone is public (see `download_urls` in the config,
or ultralytics' built-in `VisDrone.yaml` which auto-downloads).
"""

from __future__ import annotations

import shutil

import yaml

from overwatch.config import load_config, resolve


def _dataset_root(cfg):
    ds_yaml = resolve(cfg["dataset"]["yaml"])
    if not ds_yaml.exists():
        raise FileNotFoundError(
            f"dataset.yaml not found at {ds_yaml}. Point dataset.yaml at your VisDrone "
            f"YOLO export, or download VisDrone (see config download_urls)."
        )
    spec = yaml.safe_load(ds_yaml.read_text())
    root = (ds_yaml.parent / spec.get("path", ".")).resolve()
    return ds_yaml, spec, root


def verify(cfg=None) -> dict:
    cfg = cfg or load_config()
    ds_yaml, spec, root = _dataset_root(cfg)
    counts = {}
    for split in ("train", "val"):
        img_dir = root / spec.get(split, f"images/{split}")
        imgs = [p for p in img_dir.glob("*") if p.suffix.lower() in {".jpg", ".png", ".jpeg"}]
        # Resolve symlinks to confirm the underlying images actually exist.
        reachable = sum(1 for p in imgs if p.resolve().exists())
        counts[split] = {"listed": len(imgs), "reachable": reachable}
    print(f"[data] dataset.yaml: {ds_yaml}")
    print(f"[data] classes ({spec['nc']}): {spec['names']}")
    for split, c in counts.items():
        print(f"[data] {split}: {c['reachable']}/{c['listed']} images reachable")
    return {"yaml": str(ds_yaml), "root": str(root), "counts": counts, "spec": spec}


def build_sample(cfg=None) -> str:
    """Copy N val images+labels into data/sample and write a standalone sample.yaml."""
    cfg = cfg or load_config()
    _, spec, root = _dataset_root(cfg)
    n = cfg["dataset"]["sample_images"]
    out = resolve(cfg["dataset"]["sample_dir"])
    (out / "images/val").mkdir(parents=True, exist_ok=True)
    (out / "labels/val").mkdir(parents=True, exist_ok=True)

    val_imgs = sorted(p for p in (root / spec.get("val", "images/val")).glob("*")
                      if p.suffix.lower() in {".jpg", ".png", ".jpeg"})
    val_lbls_dir = root / spec.get("val", "images/val").replace("images", "labels")

    copied = 0
    for img in val_imgs:
        if copied >= n:
            break
        lbl = val_lbls_dir / f"{img.stem}.txt"
        if not lbl.exists():
            continue
        shutil.copy(img.resolve(), out / "images/val" / img.name)  # materialize (deref symlink)
        shutil.copy(lbl, out / "labels/val" / lbl.name)
        copied += 1

    sample_yaml = out / "sample.yaml"
    sample_yaml.write_text(yaml.safe_dump({
        "path": str(out), "train": "images/val", "val": "images/val",
        "nc": spec["nc"], "names": spec["names"],
    }, sort_keys=False))
    print(f"[data] wrote {copied}-image sample -> {out} ({sample_yaml})")
    return str(sample_yaml)


def main() -> int:
    cfg = load_config()
    verify(cfg)
    build_sample(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
