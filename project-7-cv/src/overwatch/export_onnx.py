"""Stage 4a — Export the fine-tuned detector to ONNX (portable, runtime-agnostic serving)."""

from __future__ import annotations

import shutil

from overwatch.config import load_config, resolve
from overwatch.evaluate import _weights


def run(cfg=None) -> str:
    from ultralytics import YOLO

    cfg = cfg or load_config()
    weights = _weights(cfg)
    print(f"[export] {weights} -> ONNX (imgsz={cfg['model']['imgsz']})")
    model = YOLO(weights)
    onnx_path = model.export(format="onnx", imgsz=cfg["model"]["imgsz"], opset=12)

    dst = resolve(cfg["serve"]["onnx"])
    dst.parent.mkdir(parents=True, exist_ok=True)
    if str(onnx_path) != str(dst):
        shutil.copy(onnx_path, dst)
    print(f"[export] wrote {dst}")
    return str(dst)


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
