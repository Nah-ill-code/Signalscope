"""
Training loop for the real-vs-AI-generated classifier.

Usage:
    python src/train.py --data-dir ./data --backbone efficientnet_b0 \
        --img-size 224 --batch-size 64 --epochs 15 --lr 3e-4 --out-dir ./checkpoints
"""
import argparse
import os
import time

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from dataset import get_dataloaders
from model import get_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--backbone", default="efficientnet_b0",
                    choices=["efficientnet_b0", "resnet50", "resnet18"])
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--out-dir", default="./checkpoints")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--init-checkpoint", default=None,
                    help="Path to an existing checkpoint (e.g. from a previous run) to "
                         "continue training from, instead of starting from ImageNet weights. "
                         "Useful for fine-tuning on a new dataset without forgetting what an "
                         "earlier dataset already taught the model.")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits)
        all_probs.extend(probs.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
    auc = roc_auc_score(all_labels, all_probs)
    return total_loss / len(loader.dataset), auc


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader, _, class_to_idx = get_dataloaders(
        args.data_dir, args.img_size, args.batch_size, num_workers=args.num_workers
    )
    print(f"class_to_idx: {class_to_idx}")
    print(f"train batches: {len(train_loader)}, val batches: {len(val_loader)}")

    model = get_model(args.backbone, pretrained=not args.no_pretrained).to(device)
    if args.init_checkpoint:
        print(f"Loading weights from {args.init_checkpoint} to fine-tune...")
        init_ckpt = torch.load(args.init_checkpoint, map_location=device)
        model.load_state_dict(init_ckpt["model_state_dict"])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_auc = 0.0
    best_path = os.path.join(args.out_dir, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(images).squeeze(1)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            pbar.set_postfix(loss=loss.item())

        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_auc = evaluate(model, val_loader, device)
        dt = time.time() - t0
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_auc={val_auc:.4f} ({dt:.1f}s)")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                "model_state_dict": model.state_dict(),
                "backbone": args.backbone,
                "img_size": args.img_size,
                "class_to_idx": class_to_idx,
                "val_auc": val_auc,
            }, best_path)
            print(f"  -> new best (val_auc={val_auc:.4f}), saved to {best_path}")

    print(f"Training complete. Best val AUC: {best_auc:.4f}. Checkpoint: {best_path}")


if __name__ == "__main__":
    main()
