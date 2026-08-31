"""Model + device helpers: a pretrained EfficientNet-B0 with its classifier head
replaced by a single-logit binary head. ImageNet features transfer well to
"does this face look statistically off" even though faces aren't ImageNet's domain —
the point of transfer learning, and a deliberately modest/standard baseline choice.
"""

from __future__ import annotations


def pick_device(requested=None) -> str:
    import torch
    if requested:
        return str(requested)
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_model(backbone: str = "efficientnet_b0"):
    import torch.nn as nn
    from torchvision import models

    if backbone != "efficientnet_b0":
        raise ValueError(f"unsupported backbone: {backbone}")
    net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = net.classifier[1].in_features
    net.classifier[1] = nn.Linear(in_features, 1)  # single logit: fake probability
    return net


def set_backbone_trainable(model, trainable: bool) -> None:
    """Freeze/unfreeze everything except the classifier head."""
    for name, p in model.named_parameters():
        if not name.startswith("classifier"):
            p.requires_grad = trainable
