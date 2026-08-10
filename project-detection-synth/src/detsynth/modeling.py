"""Shared model/device/LoRA helpers for the SFT and DPO stages."""

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


def load_tokenizer(base: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(path_or_name: str, device: str, dtype: str = "float32"):
    import torch
    from transformers import AutoModelForCausalLM
    dt = {"float32": torch.float32, "float16": torch.float16,
          "bfloat16": torch.bfloat16}[dtype]
    model = AutoModelForCausalLM.from_pretrained(path_or_name, dtype=dt)
    return model.to(device)


def lora_config(block: dict):
    from peft import LoraConfig
    return LoraConfig(
        r=block["lora_r"], lora_alpha=block["lora_alpha"],
        lora_dropout=block["lora_dropout"], target_modules=block["target_modules"],
        task_type="CAUSAL_LM", bias="none",
    )
