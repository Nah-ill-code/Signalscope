"""
Split a REAL/FAKE-labeled folder into train/test subsets in place-friendly
output dirs, so you can feed the 'train' portion into merge_datasets.py while
keeping the 'test' portion held out for honest evaluation.

Usage:
    python split_dataset.py --dir "C:/Users/darkk/OneDrive/Desktop/archive(newest)/test" --out ./archive_split --test-fraction 0.2
"""
import argparse
import os
import random
import shutil

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def find_class_dir(root: str, name: str):
    for entry in os.listdir(root):
        if entry.upper() == name.upper() and os.path.isdir(os.path.join(root, entry)):
            return os.path.join(root, entry)
    return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, help="Folder containing real/fake subfolders")
    p.add_argument("--out", required=True)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    for cls in ["REAL", "FAKE"]:
        src_dir = find_class_dir(args.dir, cls)
        if not src_dir:
            print(f"[warn] no {cls} folder found under {args.dir}")
            continue

        files = [f for f in os.listdir(src_dir) if os.path.splitext(f)[1].lower() in IMG_EXTS]
        rng.shuffle(files)
        n_test = int(len(files) * args.test_fraction)
        test_files, train_files = files[:n_test], files[n_test:]

        for split, split_files in [("train", train_files), ("test", test_files)]:
            dst_dir = os.path.join(args.out, split, cls)
            os.makedirs(dst_dir, exist_ok=True)
            for fn in split_files:
                shutil.copy2(os.path.join(src_dir, fn), os.path.join(dst_dir, fn))

        print(f"{cls}: {len(train_files)} train, {len(test_files)} test")

    print(f"\nDone. Split dataset at: {args.out}")


if __name__ == "__main__":
    main()
