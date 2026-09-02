"""Stage 2b — CLIP linear-probe: a genuinely different regime from the fine-tuned CNNs.

Frozen CLIP (ViT-B/32, OpenAI weights) visual features + a trained linear head — no
backbone fine-tuning at all. This tests a different question than train.py's CNNs:
does CLIP's web-scale vision-language pretraining already separate real/fake in its
feature space, without ANY task-specific adaptation of the backbone? That's a
meaningfully different comparison than "which fine-tuned CNN wins," not a variant of it.

Same data split as the CNN backbones (same seed → list_items/stratified_split are
deterministic) for a fair comparison, but CLIP's OWN preprocessing/normalization —
forcing it through our ImageNet-mean/std pipeline would handicap it for no reason.
"""

from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from dfdetect.config import load_config, resolve
from dfdetect.data import list_items, prepare, stratified_split
from dfdetect.metrics import binary_metrics
from dfdetect.modeling import pick_device

CLIP_ARCH = "ViT-B-32"
CLIP_PRETRAINED = "openai"


@torch.no_grad()
def _extract_features(clip_model, preprocess, items, device, batch_size=32) -> tuple[np.ndarray, np.ndarray]:
    feats, labels = [], []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        imgs = torch.stack([preprocess(Image.open(p).convert("RGB")) for p, _ in batch]).to(device)
        f = clip_model.encode_image(imgs)
        f = f / f.norm(dim=-1, keepdim=True)  # standard CLIP feature usage: L2-normalize
        feats.append(f.cpu().numpy())
        labels += [y for _, y in batch]
    return np.concatenate(feats), np.array(labels, dtype=np.float32)


def run(cfg=None, log_to_mlflow: bool = True) -> dict:
    import open_clip

    cfg = cfg or load_config()
    device = pick_device(cfg["model"]["device"])
    tcfg = cfg["train"]

    # force_quick_gelu=True: the 'openai' checkpoint was trained with QuickGELU
    # activation; open_clip's default (False) silently mismatches it, which
    # changes the actual frozen features, not just a cosmetic warning.
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_ARCH, pretrained=CLIP_PRETRAINED, device=device, force_quick_gelu=True)
    clip_model.eval()
    embed_dim = clip_model.visual.output_dim
    print(f"[clip-probe] {CLIP_ARCH}/{CLIP_PRETRAINED} loaded | embed_dim={embed_dim} | device={device}")

    paths = prepare(cfg)
    items = list_items(paths["pool"], cfg["dataset"]["max_per_class"])
    train_items, val_items = stratified_split(items, cfg["split"]["val_fraction"],
                                              cfg["split"]["random_state"])
    test_items = list_items(paths["test"])
    print(f"[clip-probe] extracting frozen features: train={len(train_items)} "
          f"val={len(val_items)} test={len(test_items)} (no backbone gradients)")

    Xtr, ytr = _extract_features(clip_model, preprocess, train_items, device)
    Xva, yva = _extract_features(clip_model, preprocess, val_items, device)
    Xte, yte = _extract_features(clip_model, preprocess, test_items, device)

    head = nn.Linear(embed_dim, 1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr_t = torch.from_numpy(Xtr).float().to(device)
    ytr_t = torch.from_numpy(ytr).float().to(device)
    Xva_t = torch.from_numpy(Xva).float().to(device)

    best_auc, best_state, patience = -1.0, None, 0
    for epoch in range(1, 31):  # linear head on cached features -> cheap, can afford more epochs
        head.train()
        opt.zero_grad()
        loss = loss_fn(head(Xtr_t).squeeze(-1), ytr_t)
        loss.backward()
        opt.step()

        head.eval()
        with torch.no_grad():
            val_scores = torch.sigmoid(head(Xva_t).squeeze(-1)).cpu().numpy()
        val_auc = binary_metrics(yva, val_scores, cfg["eval"]["threshold"])["roc_auc"]
        if val_auc > best_auc:
            best_auc, patience = val_auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        else:
            patience += 1
            if patience >= 5:
                print(f"[clip-probe] early stop at epoch {epoch} (best val AUC={best_auc:.4f})")
                break
    if epoch == 30:
        print(f"[clip-probe] completed 30 epochs (best val AUC={best_auc:.4f})")

    head.load_state_dict(best_state)
    with torch.no_grad():
        test_scores = torch.sigmoid(head(torch.from_numpy(Xte).float().to(device)).squeeze(-1)).cpu().numpy()
    metrics = binary_metrics(yte, test_scores, cfg["eval"]["threshold"])
    print(f"[clip-probe] test ROC-AUC={metrics['roc_auc']:.4f}  accuracy={metrics['accuracy']:.4f}  "
          f"F1={metrics['f1']:.4f}")

    ckpt = resolve(tcfg["ckpt_dir"]) / "clip_linear_probe"
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), ckpt / "best.pt")

    out = resolve("data/reports") / "test_eval_clip_linear_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print(f"[clip-probe] wrote {out}")

    if log_to_mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
            mlflow.set_experiment(cfg["mlflow"]["experiment"])
            with mlflow.start_run(run_name="clip_linear_probe"):
                mlflow.log_params({"backbone": "clip_vit_b32_linear_probe", "arch": CLIP_ARCH,
                                   "pretrained": CLIP_PRETRAINED})
                mlflow.log_metric("best_val_roc_auc", best_auc)
        except Exception as exc:
            print(f"[clip-probe] mlflow logging skipped: {exc}")

    return metrics


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
