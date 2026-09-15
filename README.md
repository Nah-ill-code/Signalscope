# SignalScope

Real vs. AI-generated image detector, built for the SIH 2026 "SignalScope" problem statement.

You give it an image, it tells you whether the image looks real or AI-generated, how confident it is, and *why* — with a Grad-CAM heat-map showing exactly which parts of the image influenced the decision.

## YouTube DEMO Video : https://youtu.be/xbM80QnE1G4

## Why this exists

AI-generated images are getting harder to spot with the naked eye, and most "deepfake detectors" are black boxes that just spit out a label with no reasoning. That's not good enough when the verdict actually matters — you need to be able to show *why* the model thinks something is fake, not just trust a number.

SignalScope focuses on two things: getting the classification right (measured properly with ROC-AUC and macro-F1, not just accuracy), and making the verdict explainable, so a human can sanity-check the model's reasoning instead of taking it on faith.

## Results

Evaluated on a held-out test set of 20,153 images:

| Metric | Score |
|---|---|
| ROC-AUC | 0.999 |
| Macro F1 | 0.989 |
| Accuracy @ 0.5 threshold | 98.9% |
| False positive rate | 1.1% |

Confusion matrix at the default threshold:

|  | Predicted Real | Predicted Fake |
|---|---|---|
| **Actual Real** | 9,966 | 113 |
| **Actual Fake** | 110 | 9,964 |

Full metrics, ROC curve, and confusion matrix plot are in [`report/`](./report).

**Worth being upfront about:** these numbers are on CIFAKE, where every fake image comes from a single generator (Stable Diffusion 1.4). A model that does this well on one generator can still fall apart on images from a generator it's never seen — that's the exact failure mode this problem statement is testing for. Treat this as a strong baseline, not proof the model generalizes to arbitrary AI-generated images in the wild.

## How it works

1. An image goes through a CNN backbone (transfer-learned, trained at 224×224) that outputs a real/fake probability.
2. Grad-CAM runs on the same forward pass, using gradients from the predicted class to highlight which regions of the image the model actually looked at.
3. The heat-map gets turned into a couple of plain-English bullet points (e.g. "strong activation around edges and texture boundaries") instead of just a raw image, so the explanation is actually readable.
4. Everything is wrapped in a small Gradio app so you can drag an image in and see the verdict, confidence, heat-map, and explanation together.

The model never says "this IS fake" — only a likelihood with a confidence score. Overclaiming certainty on something this probabilistic felt dishonest, and it's explicitly penalized in the rubric anyway.

## Project structure

```
signalscope/
├── app.py                  # Gradio web UI
├── requirements.txt
├── model/
│   └── best_model.pt       # trained checkpoint
├── checkpoints_v4/
│   └── best_model.pt       # trained checkpoint  that is used while running.
├── src/
│   ├── dataset.py          # data loading + transforms
│   ├── model.py             # backbone factory
│   ├── train.py              # training loop
│   ├── evaluate.py            # AUC / F1 / confusion matrix / threshold sweep
│   ├── predict.py              # single-image CLI prediction
│   ├── batch_predict.py         # batch prediction over a folder
│   ├── gradcam.py                # Grad-CAM heat-map generation
│   ├── clean_dataset.py           # dataset cleaning utilities
│   └── merge_datasets.py           # combine multiple source datasets       
└── report/                  # metrics.json, confusion_matrix.png, roc_curve.png, sample explanations
```

## Setup

```bash
# clone and enter the repo
git clone https://github.com/Nah-ill-code/Signalscope.git
cd Signalscope

# create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # macOS/Linux

# install PyTorch with CUDA first if you have a GPU — grab the right command for
# your setup from https://pytorch.org/get-started/locally/, e.g.:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# then the rest of the dependencies
pip install -r requirements.txt
```

Sanity-check the GPU is actually being picked up:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Running it

**Web app** (easiest way to try it):
```bash
python app.py --checkpoint model/best_model.pt
```
This opens a local Gradio interface — drop an image in, hit Submit, and you'll get the verdict, confidence, heat-map, and explanation side by side.

**Single-image prediction from the command line:**
```bash
python src/predict.py --checkpoint model/best_model.pt --image path/to/image.jpg
```

**Re-run evaluation on your own test set:**
```bash
python src/evaluate.py --data-dir ./data --checkpoint model/best_model.pt --img-size 224 --out-dir ./report
```

## Dataset

Trained on [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) — 100k real/AI-generated image pairs, upsampled from 32×32 to 224×224 during training. Data isn't included in this repo (too large for git); download it separately and point `--data-dir` at it if you want to retrain.

## Known limitations

- Trained fakes all come from one generator (SD1.4), so real-world generalization to newer models (Midjourney, DALL-E, Flux, etc.) is untested here.
- Explanations describe where the model looked, not a guaranteed causal reason for the verdict — Grad-CAM is an approximation, not ground truth.
- No adversarial robustness testing yet (e.g. images specifically crafted to fool the detector).

## Team

- Aryan suthar LEADER,
- Laksh Mishra
- Mudra Shukl
- Arpi Suthar
- Vaishali Vagh
- Rudrapratap Singh Chundawat

Built for Smart India Hackathon 2026.
