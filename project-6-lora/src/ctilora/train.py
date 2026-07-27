"""Stage 2 — Before/after: linear probe vs LoRA fine-tuning.

Two runs on the same encoder (DistilBERT), same data, tracked in MLflow:
  1. linear probe : freeze the encoder, train only the classification head
                    (uses the pretrained representations as-is)
  2. LoRA         : freeze the encoder weights, add low-rank adapters + head
                    (adapts the representations to the CTI domain)

We report accuracy / macro-F1 for each AND the trainable-parameter count — the
whole point of PEFT is that LoRA lifts accuracy while training ~1% of the weights.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

import mlflow
from ctilora.config import load_config, resolve


def load_splits(cfg, tokenizer):
    df = pd.read_csv(resolve(cfg["paths"]["data"]))
    maxlen = cfg["dataset"]["max_length"]

    def make(split):
        d = df[df.split == split]
        ds = Dataset.from_dict({"text": d.text.tolist(), "label": d.label.tolist()})
        return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=maxlen),
                      batched=True)

    return make("train"), make("test"), int(df.label.nunique())


def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def train_and_eval(model, train_ds, test_ds, tokenizer, cfg, tag):
    args = TrainingArguments(
        output_dir=str(resolve(f"data/ckpt_{tag}")),
        num_train_epochs=cfg["model"]["epochs"],
        per_device_train_batch_size=cfg["model"]["batch_size"],
        per_device_eval_batch_size=64,
        learning_rate=cfg["model"]["lr"],
        logging_steps=100,
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      data_collator=DataCollatorWithPadding(tokenizer))
    trainer.train()
    pred = trainer.predict(test_ds)
    y_pred = pred.predictions.argmax(-1)
    y_true = np.array(test_ds["label"])
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
    }


def main() -> int:
    cfg = load_config()
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    name = cfg["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(name)
    train_ds, test_ds, num_labels = load_splits(cfg, tokenizer)
    results = {}

    # 1) Linear probe — freeze the encoder backbone, train only the head.
    with mlflow.start_run(run_name="linear_probe"):
        # eager attention: MPS's fused SDPA can't do attention dropout during training.
        model = AutoModelForSequenceClassification.from_pretrained(
            name, num_labels=num_labels, attn_implementation="eager")
        for p in model.distilbert.parameters():
            p.requires_grad = False
        tr, total = count_params(model)
        mlflow.log_params({"method": "linear_probe", "model": name, "num_labels": num_labels})
        mlflow.log_metric("trainable_params", tr)
        m = train_and_eval(model, train_ds, test_ds, tokenizer, cfg, "probe")
        mlflow.log_metrics(m)
        results["linear_probe"] = {**m, "trainable_params": tr, "total_params": total}

    # 2) LoRA — freeze the encoder, train low-rank adapters + head.
    with mlflow.start_run(run_name="lora"):
        # eager attention: MPS's fused SDPA can't do attention dropout during training.
        model = AutoModelForSequenceClassification.from_pretrained(
            name, num_labels=num_labels, attn_implementation="eager")
        lc = cfg["lora"]
        peft_cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS, r=lc["r"], lora_alpha=lc["alpha"],
            lora_dropout=lc["dropout"], target_modules=lc["target_modules"],
            modules_to_save=["pre_classifier", "classifier"],
        )
        model = get_peft_model(model, peft_cfg)
        tr, total = count_params(model)
        mlflow.log_params({"method": "lora", "model": name, "r": lc["r"], "alpha": lc["alpha"]})
        mlflow.log_metric("trainable_params", tr)
        m = train_and_eval(model, train_ds, test_ds, tokenizer, cfg, "lora")
        mlflow.log_metrics(m)
        model.save_pretrained(str(resolve("data/lora_adapter")))
        results["lora"] = {**m, "trainable_params": tr, "total_params": total}

    reports = resolve(cfg["paths"]["reports"]); reports.mkdir(parents=True, exist_ok=True)
    (reports / "lora_results.json").write_text(json.dumps(results, indent=2))

    lp, lo = results["linear_probe"], results["lora"]
    pct = 100 * lo["trainable_params"] / lo["total_params"]
    print("\n[lora] === before/after ===")
    print(f"[lora] linear probe : acc {lp['accuracy']:.4f}  macro-F1 {lp['macro_f1']:.4f}  "
          f"({lp['trainable_params']:,} trainable)")
    print(f"[lora] LoRA         : acc {lo['accuracy']:.4f}  macro-F1 {lo['macro_f1']:.4f}  "
          f"({lo['trainable_params']:,} trainable = {pct:.2f}% of {lo['total_params']:,})")
    print(f"[lora] delta        : acc {lo['accuracy'] - lp['accuracy']:+.4f}  "
          f"macro-F1 {lo['macro_f1'] - lp['macro_f1']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
