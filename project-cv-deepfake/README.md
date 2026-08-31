# dfdetect — Deepfake / Synthetic-Face Detection

**CV learning survey, domain 1 of 8.** Fine-tunes a pretrained EfficientNet-B0 to tell real
photographs from GAN/diffusion-generated faces, on real data (not a toy/synthetic benchmark),
end-to-end: download → train → evaluate on a genuinely held-out test set → ONNX export → served
behind FastAPI → Dockerized → CI.

## Result (held-out test set, never seen during training or model selection)

| metric | value |
|---|---|
| ROC-AUC | **0.9963** |
| Accuracy | 0.9770 |
| F1 | 0.9772 |
| Confusion matrix | TN=1553 · FP=53 · FN=21 · TP=1585 (n=3212) |

Validation AUC during training peaked at 0.9985 — very close to the held-out 0.9963, which is the
sign to look for against leakage (a big drop from val→test would suggest the split leaked
information; there isn't one here, and see below for why).

## The data — and a deliberate scope cut

[pujanpaudel/deepfake_face_classification](https://huggingface.co/datasets/pujanpaudel/deepfake_face_classification)
(HF Hub, **CC BY-NC 4.0 — non-commercial**), derived from **DF40**, a published benchmark spanning
40 distinct real deepfake generation techniques (GANs + diffusion models) — not one narrow generator,
so the task is genuinely about detecting synthetic-face *artifacts* broadly, not memorizing one
model's fingerprint.

The repo ships three archives: `train.rar` (3.9GB), `val.zip` (616MB), `test.zip` (857MB). **I used
only `val.zip` + `test.zip`** (6,424 images total, ~1.47GB) and skipped `train.rar` — it's RAR format,
which needs an extra system dependency (`unrar`) this project doesn't otherwise want, and val+test
alone already gives a properly balanced, tractable training set for a genuine result. `val.zip`
becomes the **train/val pool** (re-split 80/20, stratified); `test.zip` is a **separate download**,
held out and untouched until the final evaluation — the strongest leakage guard available without a
dedicated third data source.

## An honest failure case

The model isn't perfect, and one specific example is worth showing rather than hiding: a real photo
in the sample set (the *only* `.jpg` in a pool that's otherwise almost entirely `.png`) was scored
**0.99 fake probability** — a confident, wrong prediction, part of the reported 53/1606 (3.3%) real
false-positive rate. Plausible hypothesis: **JPEG compression artifacts may superficially resemble
generation artifacts** to a model trained mostly on lossless images — the kind of format-distribution
confound that's easy to miss without checking specific failures, not just the aggregate metric.
Untested; a real next step would be evaluating format-stratified error rates directly.

## Architecture

Pretrained (ImageNet) **EfficientNet-B0**, classifier head replaced with a single logit. Two-stage
fine-tune: 1 epoch with the backbone **frozen** (warm up the new head — an untrained head sends large,
destructive gradients through pretrained weights otherwise), then **unfrozen** for the rest, early
stopping on validation ROC-AUC ([train.py](src/dfdetect/train.py)). ImageNet features transfer well
here even though faces aren't ImageNet's domain — the actual signal is low-level statistical/texture
artifacts, which generic pretrained convolutional features already pick up on.

## Serving

Exported to **ONNX** ([export_onnx.py](src/dfdetect/export_onnx.py)) and verified to match the torch
model to float32 precision (max abs diff 1.2e-7). The serving container
([docker/Dockerfile](docker/Dockerfile)) runs on **ONNX Runtime only** — no torch/torchvision — so
it's far smaller and faster to build than the training environment.

```bash
curl -F image=@photo.jpg http://localhost:8000/predict
# {"fake_probability": 0.0018, "prediction": "real"}
```

## Run it

```bash
make setup    # venv + deps
make data     # download val.zip + test.zip (~1.47GB), verify, build a tiny CI sample
make train    # fine-tune EfficientNet-B0 (early-stops on val ROC-AUC)
make eval     # score on the held-out test.zip -> data/reports/test_eval.json
make export   # -> data/dfdetect.onnx
make serve    # FastAPI /predict
make test     # split/leakage-logic tests (pure — no torch, no real download)
```

## Layout

```
conf/config.yaml         single source of truth (dataset, split, model, train, serve, MLflow)
src/dfdetect/
  data.py                download/extract/verify + pure list_items/stratified_split  [stage 1]
  dataset.py             torch Dataset wrapper (image loading, normalization)
  modeling.py             EfficientNet-B0 + head-replacement + freeze/unfreeze helpers
  train.py                two-stage fine-tune, early-stopping on val ROC-AUC          [stage 2]
  metrics.py              shared binary-classification metrics (train + eval use the same fn)
  evaluate.py              held-out test scoring                                       [stage 3]
  export_onnx.py           torch -> ONNX                                               [stage 4a]
  serve.py                 FastAPI /predict (ONNX Runtime, torch-free)                 [stage 4b]
docker/Dockerfile          torch-free serving image
tests/                     split/leakage-logic tests (pure, no model/network)
```

## The interview framing

> "I fine-tuned EfficientNet-B0 for deepfake detection on real data spanning 40 generation
> techniques, with a genuinely held-out test set — a separate download, never touched during
> training — and got 0.996 ROC-AUC. I also went looking for where it fails rather than stopping at
> the headline number: one specific real image was confidently misclassified, and it's the only
> JPEG in an otherwise-PNG pool — a plausible format-distribution confound worth investigating before
> trusting the model on JPEG-heavy real-world traffic. Served via ONNX Runtime so the deployment
> container never needs torch."

## Extensions

Format-stratified error analysis (test the JPEG-compression hypothesis directly); train on the full
pool including `train.rar` for more data; a face-forgery-localization head (which region is fake, not
just real/fake); adversarial robustness eval (does the classifier hold up against images specifically
optimized to evade it).
