"""
Scan a folder recursively for image files that are corrupted/truncated and
can't actually be loaded, and move them out of the way so training doesn't
crash on them.

Usage:
    python src/clean_dataset.py --dir "C:/Users/darkk/OneDrive/Desktop/archive(newest)"

By default this MOVES bad files into a '_corrupted' folder next to wherever
you pointed it (nothing is permanently deleted, so you can double check).
Add --delete instead to permanently delete them.
"""
import argparse
import os
import shutil

from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, help="Folder to scan recursively")
    p.add_argument("--delete", action="store_true",
                    help="Permanently delete bad files instead of moving them to _corrupted")
    return p.parse_args()


def is_valid_image(path: str) -> bool:
    try:
        with Image.open(path) as img:
            img.load()          # forces full read, catches truncation
            img.convert("RGB")  # matches what the actual dataset loader does
        return True
    except Exception:
        return False


def main():
    args = parse_args()
    root = args.dir
    quarantine_dir = os.path.join(root, "_corrupted")

    total = 0
    bad = 0
    for dirpath, _, filenames in os.walk(root):
        if dirpath.startswith(quarantine_dir):
            continue
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() not in IMG_EXTS:
                continue
            total += 1
            path = os.path.join(dirpath, fn)
            if not is_valid_image(path):
                bad += 1
                print(f"[bad] {path}")
                if args.delete:
                    os.remove(path)
                else:
                    os.makedirs(quarantine_dir, exist_ok=True)
                    dst = os.path.join(quarantine_dir, f"{bad}_{fn}")
                    shutil.move(path, dst)
            if total % 5000 == 0:
                print(f"...scanned {total} files so far ({bad} bad found)")

    print(f"\nDone. Scanned {total} images, found {bad} corrupted.")
    if bad and not args.delete:
        print(f"Bad files moved to: {quarantine_dir}")


if __name__ == "__main__":
    main()
