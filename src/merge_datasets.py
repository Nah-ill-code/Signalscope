"""
Merge additional real/AI-generated image datasets into your existing data/
folder, so train.py can use the combined, more diverse dataset without any
other code changes.

What it does:
  1. Copies your existing data/train/REAL, train/FAKE, test/REAL, test/FAKE
     into a new combined output folder (so your original data/ is untouched).
  2. Recursively finds every image file under each --real-dir you give it and
     adds it to the REAL class; every image under each --fake-dir goes to FAKE.
  3. Splits each newly-added source into train/test using --test-fraction.

Example — adding the "AiArtData vs RealArt" Kaggle dataset to your CIFAKE data:

    python src/merge_datasets.py ^
        --existing-data ./data ^
        --real-dir "C:/Users/darkk/Downloads/ai-generated-images-vs-real-images/RealArt" ^
        --fake-dir "C:/Users/darkk/Downloads/ai-generated-images-vs-real-images/AiArtData" ^
        --out ./data_combined

(On Windows PowerShell, use backtick ` instead of ^ for line continuation, or just
put it all on one line.)

Then retrain pointing at the combined folder:

    python src/train.py --data-dir ./data_combined --backbone efficientnet_b0 --img-size 224 --batch-size 64 --epochs 15 --lr 3e-4 --out-dir ./checkpoints_v2
"""
import argparse
import os
import random
import shutil

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def find_images(root: str):
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                paths.append(os.path.join(dirpath, fn))
    return paths


def copy_existing(existing_data: str, out_dir: str):
    for split in ["train", "test"]:
        for cls in ["REAL", "FAKE"]:
            src = os.path.join(existing_data, split, cls)
            dst = os.path.join(out_dir, split, cls)
            os.makedirs(dst, exist_ok=True)
            if not os.path.isdir(src):
                print(f"[warn] {src} not found, skipping")
                continue
            files = os.listdir(src)
            print(f"Copying {len(files)} existing images from {src} ...")
            for fn in files:
                s = os.path.join(src, fn)
                d = os.path.join(dst, fn)
                if not os.path.exists(d):
                    shutil.copy2(s, d)


def add_new_class_images(image_paths, cls_name: str, out_dir: str,
                          test_fraction: float, prefix: str, seed: int = 42):
    random.Random(seed).shuffle(image_paths)
    n_test = int(len(image_paths) * test_fraction)
    test_paths = image_paths[:n_test]
    train_paths = image_paths[n_test:]

    for split, paths in [("train", train_paths), ("test", test_paths)]:
        dst_dir = os.path.join(out_dir, split, cls_name)
        os.makedirs(dst_dir, exist_ok=True)
        for i, p in enumerate(paths):
            ext = os.path.splitext(p)[1].lower()
            dst = os.path.join(dst_dir, f"{prefix}_{i:06d}{ext}")
            shutil.copy2(p, dst)

    print(f"{cls_name} <- {prefix}: added {len(train_paths)} train, {len(test_paths)} test images")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--existing-data", default=None,
                    help="Path to your current data/ folder. Omit to build a combined "
                         "set from only the new sources.")
    p.add_argument("--real-dir", action="append", default=[],
                    help="Folder containing real images (searched recursively). Repeatable.")
    p.add_argument("--fake-dir", action="append", default=[],
                    help="Folder containing AI-generated images (searched recursively). Repeatable.")
    p.add_argument("--out", required=True)
    p.add_argument("--test-fraction", type=float, default=0.1)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.existing_data:
        copy_existing(args.existing_data, args.out)

    for i, d in enumerate(args.real_dir):
        imgs = find_images(d)
        print(f"Found {len(imgs)} images in {d}")
        if not imgs:
            print(f"[warn] no images found under {d} — check the path")
            continue
        add_new_class_images(imgs, "REAL", args.out, args.test_fraction, prefix=f"real_src{i}")

    for i, d in enumerate(args.fake_dir):
        imgs = find_images(d)
        print(f"Found {len(imgs)} images in {d}")
        if not imgs:
            print(f"[warn] no images found under {d} — check the path")
            continue
        add_new_class_images(imgs, "FAKE", args.out, args.test_fraction, prefix=f"fake_src{i}")

    print(f"\nDone. Combined dataset ready at: {args.out}")
    print("Next: python src/train.py --data-dir " + args.out + " --backbone efficientnet_b0 "
          "--img-size 224 --batch-size 64 --epochs 15 --lr 3e-4 --out-dir ./checkpoints_v2")


if __name__ == "__main__":
    main()
