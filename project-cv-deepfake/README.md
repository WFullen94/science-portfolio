# dfdetect — Deepfake / Synthetic-Face Detection

**CV learning survey, domain 1 of 8.** Fine-tunes and compares **4 CNN backbones + a CLIP linear
probe** to tell real photographs from GAN/diffusion-generated faces, on real data, end-to-end:
download → train → evaluate on a genuinely held-out test set → **architecture head-to-head** →
**robustness-under-distribution-shift eval** → ONNX export → served behind FastAPI → Dockerized → CI.

## Headline results

**Clean accuracy: all 4 fine-tuned CNNs are statistically tied.** Architecture barely matters once
you fine-tune a competent pretrained backbone — this is the first real finding, and it's a bit of an
anticlimax on purpose (see below for why that itself is informative).

| model | ROC-AUC | Accuracy | F1 |
|---|---|---|---|
| EfficientNet-B0 | 0.9960 | 0.9760 | 0.9763 |
| ResNet50 | 0.9957 | 0.9729 | 0.9731 |
| ConvNeXt-Tiny | 0.9945 | 0.9707 | 0.9713 |
| Xception | 0.9941 | 0.9577 | 0.9574 |
| **CLIP ViT-B/32 (linear probe, frozen)** | **0.8845** | 0.8593 | 0.8669 |

**Robustness under distribution shift is where architecture actually differentiates them** — and it
flips the "obvious" pick. Same held-out images, JPEG-recompressed / blurred / downscaled / noised
(300-image subset, [robustness_eval.py](src/dfdetect/robustness_eval.py)):

| model | clean | jpeg_q70 | jpeg_q30 | blur_r2 | downscale_4x | noise_σ15 |
|---|---|---|---|---|---|---|
| EfficientNet-B0 | 0.9990 | 0.9989 | 0.9993 | 0.9963 | 0.9973 | **0.9040** |
| **ResNet50** | 0.9999 | 0.9999 | 0.9996 | 0.9932 | 0.9917 | **0.9982** |
| ConvNeXt-Tiny | 0.9970 | 0.9969 | 0.9980 | 0.9797 | 0.9539 | 0.9969 |
| Xception | 0.9990 | 0.9986 | 0.9988 | 0.9919 | 0.9873 | 0.9986 |
| CLIP linear probe | 0.9794 | 0.9370 | 0.8794 | 0.9193 | 0.9203 | 0.8552 |

**EfficientNet-B0 — the clean-accuracy "winner" — collapses under Gaussian noise (0.999→0.904) while
ResNet50, statistically tied with it on clean data, barely moves (0.9999→0.9982).** If you only
looked at the clean leaderboard, you'd ship the wrong model. **This project now serves ResNet50**, a
decision this evidence directly drove — not the model I happened to pick first (see
[conf/config.yaml](conf/config.yaml) for the reasoning inline). The honest tradeoff: ResNet50's ONNX
export is ~94MB vs EfficientNet-B0's ~16MB — robustness costs model size here.

**The CLIP linear probe is clearly weaker AND less robust than every fine-tuned CNN**, on both
tables. This tempers a hypothesis from the design discussion — that CLIP's web-scale pretraining
might transfer surprisingly well here. It doesn't, **at least not as a bare linear probe**: CLIP's
frozen features are general-purpose (built for semantic/image-text alignment), and apparently don't
linearly encode the low-level statistical artifacts (GAN checkerboarding, diffusion noise signatures)
this task actually depends on — whereas a *fully fine-tuned* CNN reshapes its own features
specifically to catch them. A fine-tuned CLIP, or a small MLP head instead of a bare linear layer,
might close the gap; not tested here.

## Why this replaced a planned leave-one-generator-out test

The honest story matters here. The original plan was to hold out one of DF40's 40 generation
techniques entirely and measure the AUC drop — the real test of whether a detector generalizes to
*unseen* generators, not just unseen images from generators it's already seen. That test **wasn't
possible**: this HF-hosted repackaging of DF40 doesn't preserve per-technique labels (flat filenames,
no metadata — confirmed by inspection). The obvious fallback, cross-dataset generalization (train
here, test on a fully independent real/fake dataset), also wasn't practical: the cleanest candidate
found ([InfImagine/FakeImageDataset](https://huggingface.co/datasets/InfImagine/FakeImageDataset),
which includes a StyleGAN3 subset) is **717GB total**, split into 3.2GB tar segments that can't be
partially extracted; the standard smaller alternative (140k Real and Fake Faces) is Kaggle-only, and
no Kaggle credentials are configured on this machine.

So the **robustness/distribution-shift eval above is the practical substitute** — not identical to
"unseen generator," but a genuinely realistic and arguably more deployment-relevant question: does
this hold up when images get compressed, resized, or degraded in transit, which is what happens to
*every* real-world image regardless of which generator made it. It's also directly motivated by a
finding from the single-model version of this project: a real photo was confidently misclassified,
and it was the only JPEG in an otherwise-PNG pool — this eval tests that hypothesis systematically
(`jpeg_q30`) instead of leaving it as one anecdote.

## The data — and a deliberate scope cut

[pujanpaudel/deepfake_face_classification](https://huggingface.co/datasets/pujanpaudel/deepfake_face_classification)
(HF Hub, **CC BY-NC 4.0 — non-commercial**), derived from **DF40**, a published benchmark spanning 40
distinct real deepfake generation techniques (GANs + diffusion models) — not one narrow generator.

The repo ships three archives: `train.rar` (3.9GB), `val.zip` (616MB), `test.zip` (857MB). **I used
only `val.zip` + `test.zip`** (6,424 images total, ~1.47GB) and skipped `train.rar` — RAR format needs
an extra system dependency (`unrar`) this project doesn't otherwise want. `val.zip` becomes the
**train/val pool** (re-split 80/20, stratified, same seed shared across every model — the fairness
condition for the head-to-head above); `test.zip` is a **separate download**, held out and untouched
until final evaluation — the strongest leakage guard available without a dedicated third data source.

## Architecture — all 5 models, same controlled comparison

All 4 CNN backbones share **one recipe**: 224px input, ImageNet normalization, two-stage fine-tune (1
epoch backbone-frozen head warmup, then unfrozen, early-stopping on val ROC-AUC —
[train.py](src/dfdetect/train.py)/[modeling.py](src/dfdetect/modeling.py)). One documented
simplification: Xception's native/paper resolution is 299, not 224 — held at 224 like the others for
comparability, which could understate its true ceiling. The **CLIP linear probe is a genuinely
different regime**, not a variant of fine-tuning: the ViT-B/32 backbone stays completely frozen, and
only a linear head trains on its cached features
([clip_probe.py](src/dfdetect/clip_probe.py)) — included specifically to test whether general-purpose
pretraining separates real/fake *without* any task-specific backbone adaptation.

## Serving

The configured backbone (**ResNet50**, per the robustness finding) is exported to **ONNX**
([export_onnx.py](src/dfdetect/export_onnx.py)) and verified to match the torch model to float32
precision (max abs diff 1.5e-7). The serving container ([docker/Dockerfile](docker/Dockerfile)) runs
on **ONNX Runtime only** — no torch/torchvision — so it's far smaller and faster to build than the
training environment.

```bash
curl -F image=@photo.jpg http://localhost:8000/predict
# {"fake_probability": 0.0018, "prediction": "real"}
```

## Run it

```bash
make setup       # venv + deps
make data        # download val.zip + test.zip (~1.47GB), verify, build a tiny CI sample
make train       # fine-tune the configured backbone (early-stops on val ROC-AUC)
make eval        # score on the held-out test.zip -> data/reports/test_eval_<backbone>.json
make compare     # all 4 CNNs + CLIP probe, identical split -> architecture head-to-head
make robustness  # JPEG/blur/downscale/noise eval across every trained model
make export      # -> data/dfdetect.onnx (configured backbone)
make serve       # FastAPI /predict
make test        # split/leakage-logic tests (pure — no torch, no real download)
```

## Layout

```
conf/config.yaml          single source of truth (dataset, split, model, train, serve, MLflow)
src/dfdetect/
  data.py                 download/extract/verify + pure list_items/stratified_split   [stage 1]
  dataset.py              torch Dataset wrapper (image loading, normalization)
  modeling.py             4 CNN backbones (EfficientNet-B0/ResNet50/ConvNeXt-Tiny/Xception)
  train.py                two-stage fine-tune, backbone-parameterized                  [stage 2]
  clip_probe.py           CLIP ViT-B/32 linear probe (frozen features, separate regime) [stage 2b]
  metrics.py              shared binary-classification metrics (every model uses the same fn)
  evaluate.py             held-out test scoring, backbone-parameterized                [stage 3]
  compare.py              all 5 models, identical split -> architecture head-to-head    [stage 3b]
  robustness_eval.py      JPEG/blur/downscale/noise eval across every trained model     [stage 3c]
  export_onnx.py          torch -> ONNX (configured backbone)                          [stage 4a]
  serve.py                FastAPI /predict (ONNX Runtime, torch-free)                  [stage 4b]
docker/Dockerfile          torch-free serving image
tests/                     split/leakage-logic tests (pure, no model/network)
```

## The interview framing

> "I built a controlled 5-model head-to-head for deepfake detection — 4 fine-tuned CNN backbones plus
> a CLIP linear probe, identical split, identical held-out test set. Clean accuracy barely
> differentiated the CNNs (all within 0.002 ROC-AUC), which told me architecture was a second-order
> decision *for that metric*. So I built a distribution-shift robustness eval — JPEG recompression,
> blur, downscale, noise — as a practical substitute for a leave-one-generator-out test I discovered
> wasn't possible with this data. That's where the real signal was: EfficientNet-B0, the clean-AUC
> 'winner,' collapsed under Gaussian noise while ResNet50 — statistically tied on clean data — barely
> moved. I re-pointed the served model to ResNet50 on that evidence. I also tested a real hypothesis
> about CLIP transferring well here, and reported that it doesn't (at least as a bare linear probe) —
> the honest result over the more exciting-sounding one."

## Extensions

A true leave-one-generator-out test against the raw DF40 release (gated, partially video, more
setup); fine-tuning CLIP instead of a frozen linear probe, or a small MLP head, to see if either
closes the robustness/accuracy gap; a face-forgery-localization head (which region is fake, not just
real/fake); adversarial robustness (images specifically optimized to evade the classifier, as opposed
to generic distribution shift).
