"""
Batch prediction over a large folder of images (handles 20k+ easily).

Two modes, auto-detected from your --dir structure:

1. LABELED mode — if --dir contains REAL/ and FAKE/ subfolders, this also
   computes accuracy, ROC-AUC, macro-F1, and a confusion matrix (same
   metrics as evaluate.py), since you actually know the ground truth.

2. UNLABELED mode — if --dir is just a flat folder of images (no REAL/FAKE
   subfolders), it predicts each one and writes label+confidence to a CSV.
   No accuracy is computed since there's no ground truth to compare against.

Usage:
    python batch_predict.py --checkpoint ./checkpoints/best_model.pt --dir ./my_20k_photos --out predictions.csv

Runs on GPU automatically if available, in batches, with a progress bar.
Corrupted/unreadable images are skipped and logged instead of crashing the run.
"""
import argparse
import csv
import os

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from model import get_model

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def find_images(root: str):
    """Recursively find image files, returning (path, label_or_None) pairs.
    label is 'REAL'/'FAKE' if the file sits under a same-named top-level
    subfolder of --dir, else None."""
    labeled_dirs = {}
    for entry in os.listdir(root):
        full = os.path.join(root, entry)
        if os.path.isdir(full) and entry.upper() in ("REAL", "FAKE"):
            labeled_dirs[entry.upper()] = full

    items = []
    if labeled_dirs:
        for label, d in labeled_dirs.items():
            for dirpath, _, filenames in os.walk(d):
                for fn in filenames:
                    if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                        items.append((os.path.join(dirpath, fn), label))
    else:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                    items.append((os.path.join(dirpath, fn), None))
    return items


class ImageListDataset(Dataset):
    def __init__(self, items, img_size):
        self.items = items
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        try:
            img = Image.open(path).convert("RGB")
            tensor = self.tf(img)
            ok = True
        except (UnidentifiedImageError, OSError):
            tensor = torch.zeros(3, 224, 224)  # placeholder, filtered out later
            ok = False
        return tensor, path, (label or ""), ok


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dir", required=True, help="Folder of images (or folder containing REAL/FAKE subfolders)")
    p.add_argument("--out", default="predictions.csv")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = get_model(ckpt["backbone"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    img_size = ckpt["img_size"]

    class_to_idx = ckpt.get("class_to_idx", {"REAL": 0, "FAKE": 1})
    fake_idx = {k.upper(): v for k, v in class_to_idx.items()}.get("FAKE", 1)

    items = find_images(args.dir)
    if not items:
        print(f"No images found under {args.dir}")
        return
    is_labeled = items[0][1] is not None
    print(f"Found {len(items)} images. Labeled mode: {is_labeled}")

    dataset = ImageListDataset(items, img_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    rows = []
    skipped = 0

    with torch.no_grad():
        for tensors, paths, labels, oks in tqdm(loader, desc="Predicting"):
            valid_mask = oks.bool()
            if valid_mask.sum() == 0:
                skipped += (~valid_mask).sum().item()
                continue

            valid_tensors = tensors[valid_mask].to(device, non_blocking=True)
            logits = model(valid_tensors).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()

            valid_paths = [p for p, ok in zip(paths, valid_mask.tolist()) if ok]
            valid_labels = [l for l, ok in zip(labels, valid_mask.tolist()) if ok]
            skipped += (~valid_mask).sum().item()

            for path, true_label, prob in zip(valid_paths, valid_labels, probs):
                prob_fake = float(prob) if fake_idx == 1 else float(1 - prob)
                pred_label = "FAKE" if prob_fake >= args.threshold else "REAL"
                confidence = prob_fake if pred_label == "FAKE" else 1 - prob_fake
                rows.append({
                    "path": path,
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "prob_fake": round(prob_fake, 4),
                    "confidence": round(confidence, 4),
                    "correct": (true_label == pred_label) if true_label else "",
                })

    # Write CSV
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "true_label", "predicted_label",
                                                 "prob_fake", "confidence", "correct"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} predictions to {args.out}")
    if skipped:
        print(f"Skipped {skipped} unreadable/corrupted images (still counted, not predicted).")

    if is_labeled:
        n_correct = sum(1 for r in rows if r["correct"] is True)
        n_total = len(rows)
        acc = n_correct / n_total if n_total else 0.0
        print(f"\nAccuracy on labeled set: {acc:.4f} ({n_correct}/{n_total})")

        try:
            from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
            y_true = [1 if r["true_label"] == "FAKE" else 0 for r in rows]
            y_pred = [1 if r["predicted_label"] == "FAKE" else 0 for r in rows]
            y_prob = [r["prob_fake"] for r in rows]
            auc = roc_auc_score(y_true, y_prob)
            macro_f1 = f1_score(y_true, y_pred, average="macro")
            cm = confusion_matrix(y_true, y_pred)
            print(f"ROC-AUC: {auc:.4f}")
            print(f"Macro-F1: {macro_f1:.4f}")
            print(f"Confusion matrix [[TN, FP], [FN, TP]]:\n{cm}")
        except ImportError:
            print("(install scikit-learn for AUC/F1/confusion matrix)")


if __name__ == "__main__":
    main()
