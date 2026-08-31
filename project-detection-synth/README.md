# detsynth — Grounded ATT&CK Detection-Content Generator

Turns a MITRE ATT&CK technique into a **detection pack** — example procedures, telemetry to
watch, a Sigma rule starter, and pass/fail test fixtures — **grounded in ATT&CK's own detection
data** so the output is a faithful starting point a detection engineer validates, not a hallucination.

Built for a real problem: *"I can't find realistic material to base detections on."* The under-used
answer is that **ATT&CK v17's STIX bundle ships MITRE's actual detection analytics** — monitoring
prose, platforms, and tunable knobs — which almost nobody reads. This grounds generation on them.

> **Defensive / detection-engineering only.** It produces detection artifacts (logs, procedures,
> detection logic, test cases) — never functional exploit or malware code. And it's a **hypothesis
> generator, not a truth source**: synthetic telemetry must be validated against real data before you
> ship a detection on it.

## What it produces (a "detection pack")

```
technique  ─►  grounding (description · platforms · MITRE analytics + tunable knobs)  ─►
     ├─ procedures  — concrete ways the technique is carried out
     ├─ telemetry   — log/event sources, fields, command-lines, process chains
     ├─ sigma       — a Sigma rule skeleton grounded in that telemetry
     └─ fixtures    — positives that should fire, a hard-negative, a benign case
```

See a real, unedited pack: [examples/T1053.005-pack-7b.md](examples/T1053.005-pack-7b.md)
(Qwen2.5-7B, Scheduled Task) — correct Event IDs (4697/4698/4699), real `schtasks` command-lines,
`svchost.exe → taskeng.exe` chains, labeled test fixtures.

## The grounding is the point (Phase 1 — shipped)

[stix.py](src/detsynth/stix.py) walks the STIX chain most people never touch —
`analytic ←(analytic_refs)— detection-strategy —(detects)→ technique` — to attach each technique's
**real MITRE analytics** (697 techniques, 1,745 analytics). [generate.py](src/detsynth/generate.py)
feeds that grounding to the model with an explicit "don't invent platforms/log sources/event IDs"
instruction, so every artifact traces back to a real detection fact — and an *un*grounded output
(wrong platform, invented Event ID) is a **detectable** failure, not a silent one.

```bash
make setup
make ground                       # STIX -> per-technique grounding (data/techniques.jsonl)
make pack TID=T1053.005           # generate a detection pack
```

## Model size is the faithfulness lever (a real finding)

The same pack, same grounding, two model sizes:

| | Qwen2.5-1.5B | Qwen2.5-7B |
|---|---|---|
| Scheduled-task Event IDs | ❌ `4688` (Process Creation — wrong), invented | ✅ `4697/4698/4699` (correct) |
| Platform faithfulness | ❌ `scutil` (a **macOS** tool on a Windows technique) | ✅ all Windows tooling |
| Fluency | ❌ garbled/multilingual tokens | ✅ coherent |

**On a 36GB M3 Max, the 7B (fp16) runs locally and fixes the dangerous hallucinations.** For real
work, `model.base` is one config line — point it at a larger local or API model. The generator is
model-agnostic; bigger = more faithful.

## Phase 2 — DPO faithfulness alignment (run)

Preference-align a *trainable* 0.5B model to prefer telemetry faithful to **this** technique over a
**confusable** one, then measure faithfulness two ways on 60 held-out techniques. Pipeline:
constructed preference pairs — chosen = this technique's telemetry, rejected = its nearest-confusable
technique's (e.g. *MMC* vs *Regsvcs/Regasm*) — → LoRA **SFT** → LoRA **DPO** (β=0.1) → 3-way eval
([align_data.py](src/detsynth/align_data.py) · [sft.py](src/detsynth/sft.py) ·
[dpo.py](src/detsynth/dpo.py) · [evaluate.py](src/detsynth/evaluate.py)).

| model | grounding margin | faithful-rate | platform-faithful |
|---|---|---|---|
| base (0.5B) | 0.079 | 0.80 | **1.00** |
| SFT | 0.086 | 0.87 | 0.97 |
| DPO | 0.089 | 0.88 | 0.97 |

*margin = sim(gen, correct analytics) − sim(gen, confusable) — the metric DPO is aligned toward.
platform-faithful = fraction of outputs leaking no wrong-OS signal (e.g. a macOS tool on a Windows
technique) — an **independent** check, not derived from the training text.*

**The honest read — and the point of the exercise:** on the metric DPO *optimizes* (margin /
faithful-rate), it improves monotonically base → SFT → DPO. But on the **independent** platform check
there's **no gain — a slight regression** (base was already 100%). So **DPO moved the proxy it
optimizes, but the gain didn't transfer to an independent faithfulness measure** — exactly the
circularity risk the platform metric was built to expose. Combined with the 1.5B-vs-7B result above,
the defensible conclusion is that **for this task, model size dominates alignment**: a 7B fixes the
hallucinations a small model makes, while DPO on a 0.5B yields only a small proxy improvement that
doesn't robustly transfer. A larger trainable model, real (non-synthetic) preferences, and an
LLM-judge groundedness metric would all be needed to push this further — that's the honest next step,
not a rigged "DPO wins."

> **Reproducing:** DPO training needs a pinned stack — `transformers 4.46 + TRL 0.12 + peft 0.13`
> (the bleeding-edge TRL 0.29 / transformers 5.x hangs on MPS). Small batches + per-step
> `mps.empty_cache()` keep the 151k-vocab LM head within Apple-Silicon memory. See
> [requirements-align.txt](requirements-align.txt).

## Layout

```
conf/config.yaml            grounding source, generator + alignment models, generation params
src/detsynth/
  stix.py                   ATT&CK STIX -> per-technique grounding (analytics + knobs)   [phase 1]
  generate.py               grounded 4-part detection pack                                [phase 1]
  modeling.py               device / dtype / LoRA helpers
  align_data.py             faithfulness preference pairs (confusable-technique negatives) [phase 2]
  sft.py / dpo.py           LoRA SFT + DPO trainers                                        [phase 2]
  evaluate.py               grounding-margin faithfulness eval (base/SFT/DPO)             [phase 2]
examples/                   a real generated pack (committed evidence)
tests/                      STIX-grounding + formatting tests (pure, no model/network)
```

## The interview framing

> "I turned ATT&CK's under-used v17 detection analytics into a grounded generator that drafts a full
> detection pack — procedures, telemetry, a Sigma starter, and test fixtures — per technique, with
> every artifact traceable to a real MITRE detection fact so hallucinations are detectable. I showed
> model size is the dominant faithfulness lever (a 7B fixes wrong-Event-ID / wrong-OS errors a 1.5B
> makes), then ran a DPO faithfulness-alignment experiment with an *independent* eval metric — and
> found DPO improved the metric it optimizes but that gain didn't transfer to the independent check,
> which is the more defensible, less-rigged result than a clean 'DPO wins' would have been."
