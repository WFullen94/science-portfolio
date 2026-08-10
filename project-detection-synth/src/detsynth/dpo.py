"""Phase 2c — DPO: prefer faithful telemetry over the confusable technique's.

Direct Preference Optimization raises the log-prob of the correct technique's
telemetry and lowers the confusable one's, relative to the frozen SFT reference
(strength beta). No reward model, no RL loop — the preference becomes a supervised
loss. Trains a fresh LoRA on top of the merged SFT model.
"""

from __future__ import annotations

from detsynth.config import load_config, resolve
from detsynth.modeling import load_tokenizer, lora_config, pick_device

MAX_LEN = 512


def run(cfg=None, sft_model_dir: str | None = None, max_examples: int | None = None) -> str:
    from datasets import load_dataset
    from trl import DPOConfig, DPOTrainer

    cfg = cfg or load_config()
    acfg = cfg["align"]
    dcfg = acfg["dpo"]
    base = sft_model_dir or str(resolve(acfg["sft"]["out_dir"] + "_merged"))
    tok = load_tokenizer(base)
    ds = load_dataset("json", data_files=str(resolve(acfg["prefs_dir"]) / "dpo.jsonl"),
                      split="train")
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    out = resolve(dcfg["out_dir"])
    args = DPOConfig(
        output_dir=str(out), beta=dcfg["beta"], num_train_epochs=dcfg["epochs"],
        per_device_train_batch_size=dcfg["batch_size"],
        gradient_accumulation_steps=dcfg["grad_accum"], learning_rate=dcfg["lr"],
        max_length=MAX_LEN, max_prompt_length=MAX_LEN - 160,
        logging_steps=20, save_strategy="no", report_to="none", bf16=False, fp16=False,
    )
    print(f"[dpo] aligning {base} on {len(ds)} pairs | beta={dcfg['beta']} "
          f"device={pick_device(cfg['model']['device'])}")
    # ref_model=None + peft_config: the adapter-disabled model (= merged SFT) is the
    # frozen reference, so no second full model is held in memory.
    trainer = DPOTrainer(model=base, ref_model=None, args=args, train_dataset=ds,
                         processing_class=tok, peft_config=lora_config(acfg["sft"]))
    trainer.train()
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))
    print(f"[dpo] saved DPO adapter -> {out}")
    return str(out)


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
