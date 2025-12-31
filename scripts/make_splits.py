import os
import re
import random
import shutil
from pathlib import Path
from collections import defaultdict

RAW_ROOT = Path("data/raw/deepfashion")  
OUT_ROOT = Path("data/processed")

CLASSES = ["Dresses", "Graphic_Tees", "Pants", "Shorts", "Skirts"]
FRONT_PATTERN = re.compile(r"_front\.(jpg|jpeg|png)$", re.IGNORECASE)

SEED = 42
MAX_PER_CLASS = 2000          
TRAIN_FRAC = 0.7
VAL_FRAC = 0.15               

def find_front_images(class_dir: Path):
    imgs = []
    for p in class_dir.rglob("*"):
        if p.is_file() and FRONT_PATTERN.search(p.name):
            imgs.append(p)
    return imgs

def ensure_empty_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

def copy_images(paths, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in paths:
        parent = src.parent.name
        dst = dst_dir / f"{parent}__{src.name}"
        shutil.copy2(src, dst)

def main():
    random.seed(SEED)

    # reset processed folders
    ensure_empty_dir(OUT_ROOT / "train")
    ensure_empty_dir(OUT_ROOT / "val")
    ensure_empty_dir(OUT_ROOT / "test")

    counts = {}

    for cls in CLASSES:
        cls_dir = RAW_ROOT / cls
        if not cls_dir.exists():
            raise FileNotFoundError(f"Class folder not found: {cls_dir}")

        imgs = find_front_images(cls_dir)
        if len(imgs) == 0:
            raise RuntimeError(f"No front images found in {cls_dir}")

        random.shuffle(imgs)
        imgs = imgs[: min(MAX_PER_CLASS, len(imgs))]

        n = len(imgs)
        n_train = int(n * TRAIN_FRAC)
        n_val = int(n * VAL_FRAC)
        train = imgs[:n_train]
        val = imgs[n_train:n_train + n_val]
        test = imgs[n_train + n_val:]

        copy_images(train, OUT_ROOT / "train" / cls)
        copy_images(val, OUT_ROOT / "val" / cls)
        copy_images(test, OUT_ROOT / "test" / cls)

        counts[cls] = {"total": n, "train": len(train), "val": len(val), "test": len(test)}

    print("Done. Counts:")
    for k, v in counts.items():
        print(k, v)

if __name__ == "__main__":
    main()
