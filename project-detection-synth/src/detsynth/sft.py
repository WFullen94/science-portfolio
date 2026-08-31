"""Phase 2b — LoRA SFT: teach the trainable model to emit faithful telemetry.

Supervised fine-tune on (light grounding -> telemetry from the technique's real
analytics). This is the reference policy DPO aligns on top of, so we merge the
adapter and save a standalone SFT model.
"""

from __future__ import annotations

from detsynth.config import load_config, resolve
from detsynth.modeling import load_tokenizer, lora_config, mps_empty_cache_callback, pick_device

MAX_LEN = 320


def run(cfg=None, max_examples: int | None = None) -> str:
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    cfg = cfg or load_config()
    acfg = cfg["align"]
    scfg = acfg["sft"]
    tok = load_tokenizer(acfg["base"])
    ds = load_dataset("json", data_files=str(resolve(acfg["prefs_dir"]) / "sft.jsonl"),
                      split="train")
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))
    # TRL 0.12's SFTTrainer trains on a `text` field, so render the conversational
    # `messages` (prompt + gold telemetry) through the chat template up front.
    ds = ds.map(lambda ex: {"text": tok.apply_chat_template(ex["messages"], tokenize=False)},
                remove_columns=["messages"])

    out = resolve(scfg["out_dir"])
    args = SFTConfig(
        output_dir=str(out), num_train_epochs=scfg["epochs"],
        per_device_train_batch_size=scfg["batch_size"],
        gradient_accumulation_steps=scfg["grad_accum"], learning_rate=scfg["lr"],
        max_seq_length=MAX_LEN, logging_steps=1, save_strategy="no",
        report_to="none", bf16=False, fp16=False, dataloader_num_workers=0,
    )
    print(f"[sft] {acfg['base']} on {len(ds)} examples | device={pick_device(cfg['model']['device'])}")
    trainer = SFTTrainer(model=acfg["base"], args=args, train_dataset=ds,
                         processing_class=tok, peft_config=lora_config(scfg), callbacks=[mps_empty_cache_callback()])
    trainer.train()

    merged_dir = resolve(scfg["out_dir"] + "_merged")
    trainer.model.merge_and_unload().save_pretrained(str(merged_dir))
    tok.save_pretrained(str(merged_dir))
    print(f"[sft] merged SFT model -> {merged_dir}")
    return str(merged_dir)


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
