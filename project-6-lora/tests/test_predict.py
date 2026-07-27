"""Inference test — skips cleanly if the LoRA adapter hasn't been trained yet."""

import pytest

from ctilora.config import load_config, resolve

pytestmark = pytest.mark.skipif(
    not (resolve("data/lora_adapter") / "adapter_config.json").exists(),
    reason="no trained adapter yet — run `python -m ctilora.train` first",
)


def test_classify_returns_in_vocabulary_techniques():
    from ctilora.predict import classify, load_model

    cfg = load_config()
    model, tokenizer, label_map = load_model(cfg)
    out = classify(model, tokenizer, label_map,
                   "The malware used PowerShell to download and run a payload.", top_k=3)
    assert len(out) == 3
    valid_ids = {v["technique_id"] for v in label_map.values()}
    for technique_id, _name, prob in out:
        assert technique_id in valid_ids
        assert 0.0 <= prob <= 1.0
