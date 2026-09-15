"""
Evaluation script producing exactly the metrics the problem statement requires
(Section 4.2): ROC-AUC, macro-F1, confusion matrix, and accuracy/FPR at a fixed
operating threshold.

Usage:
    python src/evaluate.py --data-dir ./data --checkpoint ./checkpoints/best_model.pt \
        --img-size 224 --threshold 0.5 --out-dir ./report

To evaluate the organizers' held-out set (once you have it, including the
unseen-generator split), point --data-dir at that folder instead — it must
follow the same test/REAL, test/FAKE layout. Run it separately for the
unseen-generator subset and report that AUC too.
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (confusion_matrix, f1_score, roc_auc_score,
                              roc_curve, ConfusionMatrixDisplay)

from dataset import get_dataloaders
from model import get_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--out-dir", default="./report")
    return p.parse_args()


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images).squeeze(1)
        probs = torch.sigmoid(logits)
        all_probs.extend(probs.cpu().numpy().tolist())
        all_labels.extend(labels.numpy().tolist())
    return np.array(all_labels), np.array(all_probs)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = get_model(ckpt["backbone"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    _, _, test_loader, class_to_idx = get_dataloaders(
        args.data_dir, args.img_size, args.batch_size
    )
    print(f"class_to_idx: {class_to_idx}")

    labels, probs = run_inference(model, test_loader, device)

    # ImageFolder assigns indices alphabetically, so 'FAKE' can end up as 0 or 1
    # depending on your folder names. Normalize so that everything downstream
    # consistently treats 1 = FAKE, regardless of how the folders were indexed.
    fake_idx = {k.upper(): v for k, v in class_to_idx.items()}.get("FAKE", 1)
    if fake_idx == 0:
        labels = 1 - labels
        probs = 1 - probs

    preds = (probs >= args.threshold).astype(int)

    auc = roc_auc_score(labels, probs)
    macro_f1 = f1_score(labels, preds, average="macro")
    cm = confusion_matrix(labels, preds)

    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    fpr_at_threshold = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    metrics = {
        "roc_auc": float(auc),
        "macro_f1": float(macro_f1),
        "threshold": args.threshold,
        "accuracy_at_threshold": float(accuracy),
        "false_positive_rate_at_threshold": float(fpr_at_threshold),
        "confusion_matrix": cm.tolist(),
        "n_samples": int(len(labels)),
    }

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    # Confusion matrix plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["REAL", "FAKE"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(args.out_dir, "confusion_matrix.png"), bbox_inches="tight")
    plt.close()

    # ROC curve plot
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(args.out_dir, "roc_curve.png"), bbox_inches="tight")
    plt.close()

    print(f"\nSaved metrics.json, confusion_matrix.png, roc_curve.png to {args.out_dir}")


if __name__ == "__main__":
    main()
