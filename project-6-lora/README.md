# LoRA Fine-Tuning — CTI Procedure Text → ATT&CK Technique

Fine-tunes an encoder to classify **threat-report "procedure" text** into MITRE ATT&CK techniques,
using **LoRA** (parameter-efficient fine-tuning). The point isn't the training loop — it's the
**measured before/after** and the parameter efficiency.

## The before/after (the whole point)

Same encoder (DistilBERT), same data, two ways of adapting it:

| | Accuracy | Macro-F1 | Trainable params |
|---|---|---|---|
| **Linear probe** (freeze encoder, train head only) | 0.600 | 0.584 | 606k |
| **LoRA** (freeze encoder, add low-rank adapters) | **0.920** | **0.917** | 753k = **1.11%** of 67.7M |
| **Δ** | **+0.320** | +0.333 | — |

Adapting the encoder with LoRA — training **just 1.11% of the weights** — lifts accuracy from
60% to 92%. The adapter file is a few MB; you ship it on top of the frozen base model. Both runs are
tracked in **MLflow**.

## LoRA in one paragraph
Full fine-tuning updates all ~67M weights. LoRA freezes them and injects small trainable rank-8
adapter matrices into the attention projections; only those (+ the classifier head) train. The
weight *update* a task needs is low-rank, so `A·B` approximates it at ~1% of the parameters, ~full
quality, and a tiny per-task artifact. (**QLoRA** adds 4-bit quantization — needs CUDA, so not here.)

## The data
5,798 procedure descriptions from ATT&CK "uses" relationships (e.g. "created a scheduled task to run
the implant at logon"), labeled by technique, restricted to the **top-20 most frequent techniques**
(discovery / execution / exfiltration heavy) so classes have enough examples.

## Demo (in-scope predictions)

```bash
python -m ctilora.predict "The actor created a scheduled task to run the implant at logon."
#   0.991  T1053.005  Scheduled Task
python -m ctilora.predict "The backdoor enumerated files and directories to find documents."
#   0.956  T1083      File and Directory Discovery
```

## Run it

```bash
python -m ctilora.build_dataset   # STIX procedures -> labeled dataset (top-20 techniques)
python -m ctilora.train           # linear probe vs LoRA, tracked in MLflow
python -m ctilora.predict "..."   # classify a CTI sentence with the LoRA adapter
```

## Layout

```
conf/config.yaml       model, LoRA (r/alpha/targets), dataset, MLflow
src/ctilora/
  build_dataset.py     STIX "uses" procedures -> labeled text            [stage 1]
  train.py             linear-probe vs LoRA before/after + MLflow        [stage 2]
  predict.py           classify new CTI text with the LoRA adapter       [stage 3]
```

## The interview framing
> "I LoRA-fine-tuned an encoder to map threat-report procedure text to ATT&CK techniques, and framed
> it as a controlled before/after: a frozen-backbone linear probe (60% accuracy) vs LoRA adaptation
> (92%), where LoRA trained just 1.1% of the parameters. That isolates the value of adaptation and
> shows I understand *why* PEFT works — the update is low-rank — not just how to call a trainer."

## Extensions
Swap DistilBERT for a domain encoder (SciBERT); a decoder-LLM LoRA (generative) on a CUDA box;
full fine-tuning as an upper bound to show LoRA captures most of the gain at ~1% of the cost.
