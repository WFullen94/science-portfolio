"""Stage 6 — robustness / distribution-shift eval: the practical substitute test.

A true leave-one-generation-technique-out split isn't possible with this dataset's
packaging (no per-technique labels survived — see README), and the readily-available
cross-dataset alternatives were either impractically large (717GB) or Kaggle-gated
(no local credentials). So instead: apply real-world-like perturbations — JPEG
re-compression, blur, downscale, noise — to the SAME held-out test images and
re-score every trained model. This is directly motivated by an earlier finding: a
real image was confidently misclassified, and it was the only JPEG in an otherwise-PNG
pool. Rather than leave that as an anecdote, this tests it systematically, across
every architecture, so we can ask "does clean accuracy predict robustness" — a more
informative question than one more clean-AUC number once clean AUC is already
near-ceiling for every backbone.

Runs on a class-balanced SUBSET of the held-out test set (not all 3,212 — 5 models x
~6 conditions x the full set would be a lot of forward passes for marginal extra
precision; see `n_per_class`).
"""

from __future__ import annotations

import io
import json

import numpy as np
import torch
from PIL import Image, ImageFilter

from dfdetect import clip_probe
from dfdetect.config import load_config, resolve
from dfdetect.data import list_items, prepare
from dfdetect.dataset import IMAGENET_MEAN, IMAGENET_STD
from dfdetect.evaluate import load_trained
from dfdetect.metrics import binary_metrics
from dfdetect.modeling import BACKBONES, pick_device


def _perturb_jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _perturb_blur(img: Image.Image, radius: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius))


def _perturb_downscale(img: Image.Image, factor: int) -> Image.Image:
    w, h = img.size
    small = img.resize((max(1, w // factor), max(1, h // factor)), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def _perturb_noise(img: Image.Image, sigma: float) -> Image.Image:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    noisy = arr + np.random.default_rng(0).normal(0, sigma, arr.shape)  # fixed seed: reproducible
    return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))


PERTURBATIONS = {
    "clean": lambda img: img,
    "jpeg_q70": lambda img: _perturb_jpeg(img, 70),
    "jpeg_q30": lambda img: _perturb_jpeg(img, 30),   # the condition motivated by the JPEG finding
    "blur_r2": lambda img: _perturb_blur(img, 2),
    "downscale_4x": lambda img: _perturb_downscale(img, 4),
    "noise_sigma15": lambda img: _perturb_noise(img, 15),
}


def _cnn_score(model, pil_imgs: list[Image.Image], img_size: int, device: str) -> np.ndarray:
    arrs = []
    for img in pil_imgs:
        img = img.convert("RGB").resize((img_size, img_size))
        arr = (np.asarray(img, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        arrs.append(arr.transpose(2, 0, 1))
    x = torch.from_numpy(np.stack(arrs)).float().to(device)
    with torch.no_grad():
        logits = model(x).squeeze(-1)
    return torch.sigmoid(logits).cpu().numpy()


def _clip_score(clip_model, head, preprocess, pil_imgs: list[Image.Image], device: str) -> np.ndarray:
    x = torch.stack([preprocess(img.convert("RGB")) for img in pil_imgs]).to(device)
    with torch.no_grad():
        f = clip_model.encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)
        logits = head(f).squeeze(-1)
    return torch.sigmoid(logits).cpu().numpy()


def run(cfg=None, n_per_class: int = 150) -> dict:
    import open_clip

    cfg = cfg or load_config()
    device = pick_device(cfg["model"]["device"])
    mcfg = cfg["model"]

    paths = prepare(cfg)
    test_items = list_items(paths["test"], max_per_class=n_per_class)
    labels = np.array([y for _, y in test_items], dtype=np.float32)
    print(f"[robust] {len(test_items)}-image subset ({n_per_class}/class) of the held-out "
          f"test set, {len(PERTURBATIONS)} conditions, {len(BACKBONES) + 1} models")

    cnn_models = {bb: load_trained(bb, cfg, device) for bb in BACKBONES}

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        clip_probe.CLIP_ARCH, pretrained=clip_probe.CLIP_PRETRAINED, device=device,
        force_quick_gelu=True)  # match the openai checkpoint's actual activation
    clip_model.eval()
    clip_head = torch.nn.Linear(clip_model.visual.output_dim, 1).to(device)
    clip_ckpt = resolve(cfg["train"]["ckpt_dir"]) / "clip_linear_probe" / "best.pt"
    clip_head.load_state_dict(torch.load(clip_ckpt, map_location=device))
    clip_head.eval()

    results: dict[str, dict[str, float]] = {}
    for pname, pfn in PERTURBATIONS.items():
        imgs = [pfn(Image.open(p)) for p, _ in test_items]
        for bb, model in cnn_models.items():
            auc = binary_metrics(labels, _cnn_score(model, imgs, mcfg["img_size"], device))["roc_auc"]
            results.setdefault(bb, {})[pname] = auc
        clip_auc = binary_metrics(labels, _clip_score(
            clip_model, clip_head, clip_preprocess, imgs, device))["roc_auc"]
        results.setdefault("clip_vit_b32_linear_probe", {})[pname] = clip_auc
        print(f"[robust] {pname:<16} done")

    conds = list(PERTURBATIONS.keys())
    print("\n[robust] ROC-AUC by model x perturbation (held-out test subset):\n")
    print("| model | " + " | ".join(conds) + " |")
    print("|" + "---|" * (len(conds) + 1))
    for name, cond_scores in results.items():
        print(f"| {name:<26} | " + " | ".join(f"{cond_scores[c]:.4f}" for c in conds) + " |")

    out = resolve("data/reports") / "robustness_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[robust] wrote {out}")
    return results


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
