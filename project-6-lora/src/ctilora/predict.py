"""Stage 3 — Inference with the LoRA-adapted model.

Loads the base encoder + the trained LoRA adapter and classifies a new CTI
sentence into an ATT&CK technique. The adapter is a few MB — the point of PEFT:
ship a tiny per-task file on top of the frozen base model.
"""

from __future__ import annotations

import json
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ctilora.config import load_config, resolve


def load_model(cfg):
    name = cfg["model"]["name"]
    label_map = json.loads(resolve(cfg["paths"]["label_map"]).read_text())
    base = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=len(label_map), attn_implementation="eager")
    model = PeftModel.from_pretrained(base, str(resolve("data/lora_adapter")))
    model.eval()
    return model, AutoTokenizer.from_pretrained(name), label_map


def classify(model, tokenizer, label_map, text, top_k=3):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        probs = model(**enc).logits.softmax(-1)[0]
    top = probs.topk(top_k)
    return [(label_map[str(i.item())]["technique_id"], label_map[str(i.item())]["name"], float(p))
            for p, i in zip(top.values, top.indices)]


def main() -> int:
    cfg = load_config()
    text = " ".join(sys.argv[1:]) or \
        "The malware enumerated running processes and injected code into explorer.exe."
    model, tokenizer, label_map = load_model(cfg)
    print(f"Text: {text}\n")
    for tid, name, p in classify(model, tokenizer, label_map, text):
        print(f"  {p:.3f}  {tid:10s} {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
