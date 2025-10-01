"""
Unified preprocessing for:
  - ISIC2019
  - ODIR-5k
  - RetinaMNIST (via medmnist)

Usage for replication of paper setup:
  ISIC2019:
    python preprocess.py ISIC2019 \
      --root_dir /path/to/ISIC_2019_train \
      --image_dir ISIC_2019_Training_Input \
      --csv_path ISIC_2019_Training_Metadata.csv \
      --output_dir labeled_input \
      --filter_downsampled

  ODIR-5k:
    python preprocess.py ODIR-5k \
      --root_dir /path/to/ODIR-5k_Train \
      --image_dir preprocessed_images \
      --csv_path full_df.csv \
      --output_dir labeled_input \
      --organize_into_label_dirs

  RetinaMNIST:
    python preprocess.py RetinaMNIST \
      --out_dir /path/to/output_root \
      --rgb \
      --merge_splits \
      --flatten_train
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Optional

import pandas as pd
from tqdm import tqdm

# ---------------------------
# ISIC2019
# ---------------------------

def process_isic2019(root_dir: str,
                     image_dir: str,
                     csv_path: str,
                     output_dir: str,
                     filter_downsampled: bool = True):
    """
    Rename/move images into a flat folder with "label_imageid.jpg".
    """
    root = Path(root_dir)
    img_dir = (root / image_dir).resolve()
    csv_fp = (root / csv_path).resolve()
    out_dir = (root / output_dir).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_fp)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="ISIC2019"):
        image_name = row["image"]
        label = row["class_id"]

        image_path = img_dir / f"{image_name}.jpg"
        if not image_path.exists():
            continue

        # Remove images containing "downsampled" if requested
        if filter_downsampled and "downsampled" in str(image_name).lower():
            try:
                image_path.unlink()
            except OSError:
                pass
            continue

        new_filename = f"{label}_{image_name}.jpg"
        new_path = out_dir / new_filename
        shutil.move(str(image_path), str(new_path))

    print(f"[ISIC2019] Done. Renamed images at: {out_dir}")


def isic_create_sampled_dataset(source_dir: str,
                                output_dir: str,
                                sample_fraction: float = 0.5,
                                seed: int = 42):
    """
    Sample a fraction of images per label from a flat folder with filenames like '<label>_<rest>.jpg'.
    """
    import random
    random.seed(seed)

    if not (0.0 <= sample_fraction <= 1.0):
        raise ValueError("sample_fraction must be between 0.0 and 1.0")

    src = Path(source_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    by_label = defaultdict(list)
    for fname in os.listdir(src):
        fpath = src / fname
        if not fpath.is_file():
            continue
        if "_" not in fname:
            continue
        lower = fname.lower()
        if not (lower.endswith(".jpg") or lower.endswith(".jpeg")):
            continue
        label = fname.split("_", 1)[0]
        by_label[label].append(fname)

    for label, files in by_label.items():
        total = len(files)
        if total == 0:
            continue
        k = int(total * sample_fraction)
        random.shuffle(files)
        selected = files[:k]

        target_dir = out / label
        target_dir.mkdir(parents=True, exist_ok=True)

        for fname in selected:
            src_fp = src / fname
            dst_fp = target_dir / fname
            shutil.copy2(src_fp, dst_fp)

        print(f"[ISIC2019] Label '{label}': copied {len(selected)} / {total}")

# ---------------------------
# ODIR-5k
# ---------------------------

def process_odir5k(root_dir: str,
                   image_dir: str,
                   csv_path: str,
                   output_dir: str,
                   organize_into_label_dirs: bool = True):
    """
    Rename/move images to 'label_filename.jpg', then group into per-label folders.
    """
    root = Path(root_dir)
    img_dir = (root / image_dir).resolve()
    csv_fp = (root / csv_path).resolve()
    out_dir = (root / output_dir).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_fp)

    label_to_id = {
        "N": 0, "D": 1, "G": 2, "C": 3,
        "A": 4, "H": 5, "M": 6, "O": 7
    }

    for _, row in tqdm(df.iterrows(), total=len(df), desc="ODIR-5k"):
        image_name = row["filename"]
        label_name = row["labels"][2]  # Preserve original behavior
        if label_name not in label_to_id:
            # Skip unknown labels
            continue
        label = label_to_id[label_name]

        image_path = img_dir / image_name
        if not image_path.exists():
            continue

        new_filename = f"{label}_{image_name}.jpg"
        new_path = out_dir / new_filename
        shutil.move(str(image_path), str(new_path))

    print(f"[ODIR-5k] Renamed images at: {out_dir}")

    if organize_into_label_dirs:
        for filename in os.listdir(out_dir):
            src_fp = out_dir / filename
            if not src_fp.is_file():
                continue
            if "_" not in filename:
                continue
            label = filename.split("_")[0]
            dest_dir = out_dir / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_fp), str(dest_dir / filename))
        print(f"[ODIR-5k] Organized into per-label directories under: {out_dir}")

# ---------------------------
# RetinaMNIST (via medmnist)
# ---------------------------

def _ensure_uint8(img):
    import numpy as np
    # Move channels to last if looks like CxHxW
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    # Remove trailing singleton channel
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[..., 0]
    if img.dtype != np.uint8:
        maxv = float(img.max()) if img.size > 0 else 1.0
        if maxv <= 1.0:
            img = (img * 255.0).round().astype("uint8")
        else:
            import numpy as np
            img = np.clip(img, 0, 255).astype("uint8")
    return img

def _label_to_int(lbl):
    import numpy as np
    lbl = np.array(lbl)
    if lbl.ndim == 0:
        return int(lbl)
    if lbl.ndim == 1 and lbl.size == 1:
        return int(lbl[0])
    return int(np.argmax(lbl))

def retina_save_split_to_images(dataset, out_root: Path, split_name: str, rgb: bool):
    """
    Save a MedMNIST split to out_root/split_name/<class>/idx.png
    """
    from PIL import Image
    imgs = dataset.imgs
    labels = dataset.labels

    split_dir = out_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    num_samples = imgs.shape[0]
    for i in range(num_samples):
        img = imgs[i]
        lbl = _label_to_int(labels[i])
        img = _ensure_uint8(img)

        if img.ndim == 2:
            mode = "L"
        elif img.ndim == 3 and img.shape[-1] == 3:
            mode = "RGB" if rgb else "RGB"
        else:
            if img.ndim == 3 and img.shape[-1] == 1:
                img = img[..., 0]
                mode = "L"
            else:
                img = img.squeeze()
                mode = "L" if img.ndim == 2 else "RGB"

        pil_img = Image.fromarray(img, mode=mode)
        class_dir = split_dir / f"{lbl}"
        class_dir.mkdir(parents=True, exist_ok=True)
        pil_img.save(class_dir / f"{i}.png")

def retina_merge_splits(root_dir: str, out_dir: Optional[str] = None):
    """
    Merge train/val/test into a single train/ folder (move files) and delete val/test.
    """
    root = Path(root_dir)
    out_root = Path(out_dir) if out_dir else root

    train_dir = out_root / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        split_dir = root / split
        if not split_dir.exists():
            print(f"Skipping {split_dir} (not found)")
            continue

        print(f"Merging {split} -> train")
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            target_class_dir = train_dir / class_dir.name
            target_class_dir.mkdir(parents=True, exist_ok=True)
            for file in class_dir.iterdir():
                if file.is_file():
                    new_name = f"{split}_{file.name}"
                    shutil.move(str(file), str(target_class_dir / new_name))
        shutil.rmtree(split_dir)
        print(f"Deleted {split_dir}")

    print(f"[RetinaMNIST] Merged dataset at: {train_dir}")

def retina_move_classes_up(root_dir: str):
    """
    Move class folders from root/train/* -> root/* and remove train/.
    """
    root = Path(root_dir)
    train_dir = root / "train"
    if not train_dir.exists():
        print(f"{train_dir} not found.")
        return
    for class_dir in train_dir.iterdir():
        if class_dir.is_dir():
            target = root / class_dir.name
            if target.exists():
                print(f"Target {target} already exists, skipping.")
                continue
            shutil.move(str(class_dir), str(target))
            print(f"Moved {class_dir} -> {target}")
    shutil.rmtree(train_dir)
    print(f"Deleted {train_dir}")

def process_retinamnist(out_dir: str,
                        rgb: bool = True,
                        merge_splits: bool = True,
                        flatten_train: bool = True):
    """
    Download (via medmnist) and export RetinaMNIST train/val/test to PNGs in class folders.
    Then optionally merge splits and/or flatten train/.
    """
    from medmnist import RetinaMNIST

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Load datasets
    train_ds = RetinaMNIST(split="train", download=True)
    val_ds   = RetinaMNIST(split="val", download=True)
    test_ds  = RetinaMNIST(split="test", download=True)

    # Save splits
    retina_save_split_to_images(train_ds, out_root, "train", rgb=rgb)
    retina_save_split_to_images(val_ds,   out_root, "val",   rgb=rgb)
    retina_save_split_to_images(test_ds,  out_root, "test",  rgb=rgb)

    if merge_splits:
        retina_merge_splits(str(out_root), out_dir=None)

    if flatten_train:
        # Move class dirs up from train/ to out_root/
        retina_move_classes_up(str(out_root))

    print(f"[RetinaMNIST] Done at: {out_root}")

# ---------------------------
# CLI
# ---------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Preprocess ISIC2019 / ODIR-5k / RetinaMNIST datasets."
    )
    subparsers = p.add_subparsers(dest="dataset", required=True)

    # ISIC2019
    p_isic = subparsers.add_parser("ISIC2019", help="Preprocess ISIC2019")
    p_isic.add_argument("--root_dir", required=True, type=str,
                        help="Path to ISIC2019 root (contains images & CSV).")
    p_isic.add_argument("--image_dir", required=True, type=str,
                        help="Relative path under root to the images folder (e.g., ISIC_2019_Training_Input).")
    p_isic.add_argument("--csv_path", required=True, type=str,
                        help="Relative path under root to metadata CSV (e.g., ISIC_2019_Training_Metadata.csv).")
    p_isic.add_argument("--output_dir", required=True, type=str,
                        help="Relative path under root for output folder (e.g., labeled_input).")
    p_isic.add_argument("--filter_downsampled", action="store_true",
                        help="If set, remove files whose names contain 'downsampled'.")

    # Optional sampling helper for ISIC
    p_isic.add_argument("--sample_from", type=str, default=None,
                        help="(Optional) Source flat folder to sample from (files named '<label>_*.jpg').")
    p_isic.add_argument("--sample_out", type=str, default=None,
                        help="(Optional) Output folder for sampled subset.")
    p_isic.add_argument("--sample_fraction", type=float, default=0.5,
                        help="(Optional) Fraction per label to sample when using sampling helper.")
    p_isic.add_argument("--sample_seed", type=int, default=42,
                        help="(Optional) RNG seed for sampling.")

    # ODIR-5k
    p_odir = subparsers.add_parser("ODIR-5k", help="Preprocess ODIR-5k")
    p_odir.add_argument("--root_dir", required=True, type=str,
                        help="Path to ODIR-5k root.")
    p_odir.add_argument("--image_dir", required=True, type=str,
                        help="Relative path under root to the images folder (e.g., preprocessed_images).")
    p_odir.add_argument("--csv_path", required=True, type=str,
                        help="Relative path under root to metadata CSV (e.g., full_df.csv).")
    p_odir.add_argument("--output_dir", required=True, type=str,
                        help="Relative path under root for output folder (e.g., labeled_input).")
    p_odir.add_argument("--organize_into_label_dirs", action="store_true",
                        help="If set, move images into subfolders per label under output_dir.")

    # RetinaMNIST
    p_ret = subparsers.add_parser("RetinaMNIST", help="Export RetinaMNIST via medmnist")
    p_ret.add_argument("--out_dir", required=True, type=str,
                       help="Folder to save exported PNGs and class folders.")
    p_ret.add_argument("--rgb", action="store_true",
                       help="If set, keep RGB mode (default True).")
    p_ret.add_argument("--merge_splits", action="store_true",
                       help="If set, merge train/val/test into a single train/ folder.")
    p_ret.add_argument("--flatten_train", action="store_true",
                       help="If set, move class folders up from train/ to out_dir/ and delete train/.")

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.dataset == "ISIC2019":
        process_isic2019(
            root_dir=args.root_dir,
            image_dir=args.image_dir,
            csv_path=args.csv_path,
            output_dir=args.output_dir,
            filter_downsampled=args.filter_downsampled,
        )
        # Optional sampling pass
        if args.sample_from and args.sample_out:
            isic_create_sampled_dataset(
                source_dir=args.sample_from,
                output_dir=args.sample_out,
                sample_fraction=args.sample_fraction,
                seed=args.sample_seed
            )

    elif args.dataset == "ODIR-5k":
        process_odir5k(
            root_dir=args.root_dir,
            image_dir=args.image_dir,
            csv_path=args.csv_path,
            output_dir=args.output_dir,
            organize_into_label_dirs=args.organize_into_label_dirs
        )

    elif args.dataset == "RetinaMNIST":
        process_retinamnist(
            out_dir=args.out_dir,
            rgb=bool(args.rgb) or True,  # default True
            merge_splits=bool(args.merge_splits),
            flatten_train=bool(args.flatten_train)
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

if __name__ == "__main__":
    main()
