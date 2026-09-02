"""Stage 4a — Export the fine-tuned classifier to ONNX for portable, torch-free serving."""

from __future__ import annotations

import torch

from dfdetect.config import load_config, resolve
from dfdetect.modeling import build_model


def run(cfg=None, backbone: str | None = None) -> str:
    cfg = cfg or load_config()
    mcfg, tcfg = cfg["model"], cfg["train"]
    backbone = backbone or mcfg["backbone"]
    model = build_model(backbone)
    ckpt = resolve(tcfg["ckpt_dir"]) / backbone / "best.pt"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 3, mcfg["img_size"], mcfg["img_size"])
    out = resolve(cfg["serve"]["onnx"])
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(out), input_names=["image"], output_names=["logit"],
        dynamic_axes={"image": {0: "batch"}, "logit": {0: "batch"}}, opset_version=17,
        dynamo=False,  # legacy TorchScript-based exporter — dynamo needs onnxscript
    )
    print(f"[export] wrote {out}")
    return str(out)


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
