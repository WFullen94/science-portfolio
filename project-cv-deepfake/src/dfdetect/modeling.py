"""Model + device helpers: fine-tuned CNN backbones with a single-logit binary head.

All backbones share ONE controlled comparison: same 224px input, same ImageNet
normalization, same freeze-then-unfreeze fine-tune recipe, same data split. That's
what makes the head-to-head in compare.py fair. The one documented exception is
Xception, whose native/paper resolution is 299 — holding it at 224 like the others is
a deliberate simplification for comparability, and could understate its ceiling (see
README). CLIP is NOT in this fine-tuning family — it gets its own linear-probe path
in clip_probe.py, because "frozen features + linear head" is a genuinely different
regime, not a variant of fine-tuning.
"""

from __future__ import annotations

BACKBONES = ["efficientnet_b0", "resnet50", "convnext_tiny", "xception"]


def pick_device(requested=None) -> str:
    import torch
    if requested:
        return str(requested)
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_model(backbone: str):
    """Return a pretrained backbone with its classifier replaced by a single logit."""
    import torch.nn as nn

    if backbone == "efficientnet_b0":
        from torchvision import models
        net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, 1)
        head_prefix = "classifier"
    elif backbone == "resnet50":
        from torchvision import models
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        net.fc = nn.Linear(net.fc.in_features, 1)
        head_prefix = "fc"
    elif backbone == "convnext_tiny":
        from torchvision import models
        net = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        net.classifier[2] = nn.Linear(net.classifier[2].in_features, 1)
        head_prefix = "classifier.2"
    elif backbone == "xception":
        import timm
        net = timm.create_model("legacy_xception", pretrained=True, num_classes=1)
        head_prefix = "fc"
    else:
        raise ValueError(f"unsupported backbone: {backbone}")

    net._head_prefix = head_prefix  # stash for set_backbone_trainable
    return net


def set_backbone_trainable(model, trainable: bool) -> None:
    """Freeze/unfreeze everything except the classifier head."""
    prefix = model._head_prefix
    for name, p in model.named_parameters():
        if not name.startswith(prefix):
            p.requires_grad = trainable
