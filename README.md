# SignalScope — Real vs AI-Generated Image Detector

A from-scratch pipeline for the SIH-2026 "SignalScope" problem statement:
classify an image as **real** or **AI-generated**, report ROC-AUC / macro-F1
on a held-out set (including unseen generators), and explain each verdict
with a Grad-CAM heat-map (Bonus Module A).

## 1. What's in this repo

```
signalscope/
├── README.md                <- you are here
├── requirements.txt
├── src/
│   ├── dataset.py            <- data loading + transforms
│   ├── model.py               <- backbone factory (transfer learning)
│   ├── train.py               <- training loop (AMP, cosine LR, checkpoints)
│   ├── evaluate.py             <- AUC / macro-F1 / confusion matrix / threshold sweep
│   ├── gradcam.py              <- Bonus A: faithful heat-map explanations
│   └── predict.py               <- single-image CLI predict interface (Section 4.1)
├── app.py                        <- minimal Gradio web UI (drag & drop)
└── report/                        <- put your one-page model report here (7.3)
```

## 2. Environment setup (RTX 5050, Windows/Linux)

```bash
# 1. Create an isolated environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install PyTorch WITH CUDA support first — go to https://pytorch.org/get-started/locally/
# and copy the exact command for your CUDA version. RTX 5050 (Blackwell) needs a recent
# CUDA build — as of writing, CUDA 12.4+ wheels. Example:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. Install the rest
pip install -r requirements.txt

# 4. Sanity check the GPU is visible
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If `torch.cuda.is_available()` prints `False`, the CUDA wheel doesn't match your driver —
reinstall torch using the exact selector from pytorch.org for your installed CUDA/driver version.

## 3. Get the data

Download the CIFAKE dataset (Kaggle: `birdy654/cifake-real-and-ai-generated-synthetic-images`)
and unzip it so you end up with:

```
data/
├── train/
│   ├── REAL/   (50,000 images, 32x32)
│   └── FAKE/   (50,000 images, 32x32, Stable Diffusion 1.4 generated)
└── test/
    ├── REAL/   (10,000 images)
    └── FAKE/   (10,000 images)
```

This structure is already `ImageFolder`-compatible — no custom parsing needed.

### Important caveat you must handle (and mention in your report)

CIFAKE images are only **32×32 px**, and all the fakes come from **one generator**
(Stable Diffusion 1.4 / CIFAR10 real photos). That means:

- A model trained only on CIFAKE will **overfit to SD1.4's specific artefacts** and
  the low resolution. It is exactly the failure mode the problem statement warns about
  ("detectors that ace one generator often fail on a new one").
- The organizers' held-out set will likely contain **higher-resolution, real-world-sized
  images** from generators CIFAKE never saw. If you only train on 32×32 CIFAKE, your
  unseen-generator AUC will suffer.

**What to do about it (recommended, and matches the problem statement's data rules
in Section 4.1 — "you may add other public synthetic-image datasets, e.g. GenImage"):**

1. Use CIFAKE as your base/starting point — it's clean, balanced, and fast to iterate on.
2. Add a second public dataset with higher-resolution, multi-generator images
   (e.g. **GenImage**, or a subset of **ArtiFact** / other public real-vs-AI sets) — cite
   whatever you use in the README per Section 7.2.
3. Train your backbone at a resolution like 224×224 (upsampling the 32×32 CIFAKE images is
   fine — it still teaches texture/frequency artefacts) so the same model generalizes to the
   organizers' likely-larger held-out images.
4. Use strong augmentation (JPEG re-compression, resize/re-upsample, slight blur) during
   training — this is also literally Bonus Module C (robustness to degradation), so you get
   it "for free."

If you only have time for CIFAKE alone, that's a legitimate MVP — just be honest about
this limitation in your report's "Limitations" field (Section 7.3), which is scored.

## 4. Train

```bash
python src/train.py \
  --data-dir ./data \
  --backbone efficientnet_b0 \
  --img-size 224 \
  --batch-size 64 \
  --epochs 15 \
  --lr 3e-4 \
  --out-dir ./checkpoints
```

Notes for your GPU:
- `efficientnet_b0` or `resnet50` are good default transfer-learning backbones — small
  enough to train quickly on a laptop GPU, strong enough to beat a from-scratch CNN baseline.
- The script uses **mixed precision (AMP)** automatically when CUDA is available, which on
  an RTX 5050 roughly doubles throughput and halves VRAM use — you should comfortably fit
  batch size 64–128 at 224×224.
- If you get an out-of-memory error, lower `--batch-size` to 32 first before lowering
  `--img-size`.
- Training holds out 10% of the CIFAKE train split as validation automatically (never
  touches `test/`, which stands in for the organizers' held-out set until you get the real one).

## 5. Evaluate (Section 4.2 — required metrics)

```bash
python src/evaluate.py \
  --data-dir ./data \
  --checkpoint ./checkpoints/best_model.pt \
  --img-size 224 \
  --threshold 0.5 \
  --out-dir ./report
```

This produces, in `./report/`:
- `metrics.json` — overall ROC-AUC, macro-F1, accuracy & FPR at your chosen threshold
- `confusion_matrix.png`
- `roc_curve.png`

When you get the organizers' actual held-out set (with the unseen-generator split), re-run
this script pointed at that folder and **report both the overall AUC and the unseen-split
AUC separately** — the unseen-split number is what's weighted most heavily (Section 4.2/9).

## 6. Explain a verdict (Bonus Module A)

```bash
python src/gradcam.py \
  --checkpoint ./checkpoints/best_model.pt \
  --image path/to/some_image.jpg \
  --img-size 224 \
  --out heatmap.png
```

This overlays a Grad-CAM heat-map on the image showing which regions most influenced the
verdict — pair this with a short natural-language description (e.g. "high activation around
the edges of the object, consistent with texture inconsistency" — describe what you actually
see in the heatmap, don't invent artefacts, or you'll lose points on the "no over-claiming"
rubric in Section 4.3).

## 7. Run a single prediction (Section 4.1 required interface)

```bash
python src/predict.py --checkpoint ./checkpoints/best_model.pt --image path/to/image.jpg
```

Outputs a JSON like:
```json
{"label": "AI-generated", "confidence": 0.88, "raw_score": 0.88}
```

## 8. Minimal web interface (also satisfies Bonus F partially)

```bash
python app.py --checkpoint ./checkpoints/best_model.pt
```

Opens a local Gradio app: drag an image in, get the verdict, confidence, and heat-map.

## 9. Suggested week plan

1. **Day 1–2:** Environment + CIFAKE baseline (ResNet/EfficientNet transfer learning,
   224×224, no augmentation beyond flips). Get a working train → evaluate → predict loop.
2. **Day 3:** Add augmentation (JPEG compression, resize jitter) + class-weighted loss if
   needed. Re-evaluate. This is your Bonus C groundwork.
3. **Day 4:** Grad-CAM explanations (Bonus A) — the headline bonus, worth the most points.
4. **Day 5:** Add a second public dataset for generator diversity if time allows; re-train;
   compare unseen-generator performance.
5. **Day 6:** Build the Gradio app, write the one-page model report, record the demo video.
6. **Day 7 buffer:** Clean up repo structure to match Section 7.1, write README per 7.2,
   double check reproducibility from a clean clone.

## 10. Common pitfalls that cost points

- Training/tuning on what will become your "test" set (leakage) — keep them separate always.
- Reporting only accuracy, not AUC + macro-F1 + confusion matrix (all required).
- Claiming certainty ("this IS fake") instead of a likelihood — the rubric explicitly
  penalizes over-claiming.
- A repo that doesn't run cleanly from the README on a judge's machine — reproducibility is
  a scored gate (Section 9), test this yourself on a fresh clone before submitting.
