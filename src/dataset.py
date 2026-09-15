"""
Data loading for the real-vs-AI-generated classification task.

Expects an ImageFolder-compatible layout:
    data_dir/train/REAL/*.png
    data_dir/train/FAKE/*.png
    data_dir/test/REAL/*.png
    data_dir/test/FAKE/*.png

Label convention (fixed everywhere in this repo):
    0 = REAL
    1 = FAKE (AI-generated)
"""
import random
from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

CLASS_TO_IDX_EXPECTED = {"REAL": 0, "FAKE": 1}


class JPEGCompress:
    """Randomly re-compress the image as JPEG to simulate real-world degradation.
    This is cheap augmentation that also builds toward Bonus Module C
    (robustness to compression)."""

    def __init__(self, quality_range=(30, 90), p=0.3):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        quality = random.randint(*self.quality_range)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class RandomLightBlur:
    """Occasionally blur slightly, simulating screenshots / re-saves."""

    def __init__(self, p=0.15):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        radius = random.uniform(0.3, 1.2)
        return img.filter(ImageFilter.GaussianBlur(radius))


def build_transforms(img_size: int, train: bool):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            RandomLightBlur(p=0.15),
            JPEGCompress(p=0.3),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])


def get_dataloaders(data_dir: str, img_size: int, batch_size: int,
                     val_fraction: float = 0.1, num_workers: int = 4, seed: int = 42):
    """Returns train_loader, val_loader, test_loader, class_to_idx.

    train/ is split into train+val internally (val_fraction held out).
    test/ is used only for final evaluation.
    """
    train_tf = build_transforms(img_size, train=True)
    eval_tf = build_transforms(img_size, train=False)

    full_train = datasets.ImageFolder(f"{data_dir}/train", transform=train_tf)
    class_to_idx = full_train.class_to_idx
    if class_to_idx != CLASS_TO_IDX_EXPECTED:
        print(f"[warn] class_to_idx is {class_to_idx}, expected {CLASS_TO_IDX_EXPECTED}. "
              f"Metrics/labels downstream assume FAKE=1, REAL=0 — double check your folder names.")

    n_val = int(len(full_train) * val_fraction)
    n_train = len(full_train) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [n_train, n_val], generator=generator)

    # val subset should use eval transforms (no augmentation), not train transforms
    val_set.dataset = datasets.ImageFolder(f"{data_dir}/train", transform=eval_tf)

    test_set = datasets.ImageFolder(f"{data_dir}/test", transform=eval_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, class_to_idx
