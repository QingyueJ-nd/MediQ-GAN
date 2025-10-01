"""
LPIPS Intra-class Diversity Evaluation

Computes intra-class LPIPS diversity scores for generated images.
Supports both standard LPIPS backbones (alex, vgg, squeeze) and custom ResNet18 models.

Usage with standard LPIPS:
  python eval_lpips.py \
    --real_dir /path/to/real/images \
    --gen_dir /path/to/generated/images \
    --output_json results/lpips_results.json \
    --lpips_backbone alex \
    --image_size 64

Usage with custom ResNet18 model:
  python eval_lpips.py \
    --real_dir /path/to/real/images \
    --gen_dir /path/to/generated/images \
    --output_json results/lpips_results.json \
    --custom_model_path models/resnet18_lpips.pth \
    --image_size 64
"""

import os
import sys
import json
import random
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")

try:
    import lpips
except ImportError:
    lpips = None


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    """Create parent directories if they don't exist."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


class ImageFolderDataset(Dataset):
    """
    Loads images from root with class subdirectories:
      root/
        class_a/*.png
        class_b/*.png
    
    Returns: (image_tensor, class_index, class_name, image_path)
    """
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(self, root_dir: str, transform: Optional[T.Compose] = None):
        self.root = Path(root_dir)
        self.transform = transform
        self.samples: List[Tuple[str, int, str]] = []
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}

        class_dirs = sorted([d for d in self.root.iterdir() if d.is_dir()])
        for idx, cdir in enumerate(class_dirs):
            cname = cdir.name
            self.class_to_idx[cname] = idx
            self.idx_to_class[idx] = cname
            for p in cdir.iterdir():
                if p.suffix.lower() in self.IMG_EXTS:
                    self.samples.append((str(p), idx, cname))

        if len(self.samples) == 0:
            raise ValueError(f"No images found in {root_dir}. Check directory structure.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i: int):
        path, cidx, cname = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, cidx, cname, path


def load_custom_model(model_path: str, device: str = "cuda"):
    """
    Load custom ResNet18-based perceptual model.
    
    Expected model structure:
      - Must have embed() method that takes input tensor and returns embeddings
      - Must have pair_distance() method that takes two embeddings and returns distance
    
    Args:
        model_path: Path to model checkpoint (.pth file)
        device: Device to load model on
    
    Returns:
        Tuple of (model, is_custom=True, input_mode='imagenet')
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Add the model directory to path to allow imports
    model_dir = os.path.dirname(os.path.abspath(model_path))
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    
    try:
        # Try to import the model class
        # Assumes the model file defines ResNet18PerceptualNet class
        from lpips_trainer import ResNet18PerceptualNet
        
        print(f"Loading custom ResNet18 model from {model_path}")
        ckpt = torch.load(model_path, map_location=dev)
        model = ResNet18PerceptualNet(use_multi_scale=True).to(dev).eval()
        model.load_state_dict(ckpt["model_state_dict"])
        print("Custom ResNet18 model loaded successfully")
        return model, True, "imagenet"
    except Exception as e:
        raise RuntimeError(f"Failed to load custom model from {model_path}: {e}")


def load_standard_lpips(backbone: str = "alex", device: str = "cuda"):
    """
    Load standard LPIPS model.
    
    Args:
        backbone: One of 'alex', 'vgg', 'squeeze'
        device: Device to load model on
    
    Returns:
        Tuple of (model, is_custom=False, input_mode='lpips')
    """
    if lpips is None:
        raise ImportError("lpips package required. Install with: pip install lpips")
    
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    net = lpips.LPIPS(net=backbone).to(dev).eval()
    print(f"Using standard LPIPS backbone: {backbone}")
    return net, False, "lpips"


def build_lpips_transforms(mode: str = "lpips", image_size: int = 64) -> T.Compose:
    """
    Build transforms for LPIPS evaluation.
    
    Args:
        mode: 'imagenet' for custom ResNet18 models, 'lpips' for standard LPIPS
        image_size: Image resolution
    
    Returns:
        Transform composition
    """
    if mode == "imagenet":
        # ImageNet normalization for custom ResNet18 models
        return T.Compose([
            T.Resize(image_size, antialias=True),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
        ])
    elif mode == "lpips":
        # Standard LPIPS expects [-1, 1] range
        return T.Compose([
            T.Resize(image_size, antialias=True),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Lambda(lambda x: x * 2.0 - 1.0)  # [0,1] -> [-1,1]
        ])
    else:
        raise ValueError("mode must be 'imagenet' or 'lpips'")


@torch.no_grad()
def compute_intra_class_lpips(
    class_to_paths: Dict[str, List[str]],
    lpips_model: nn.Module,
    is_custom: bool,
    input_mode: str,
    device: str,
    image_size: int,
    max_pairs_per_class: int = 200,
    all_pairs_threshold: int = 50,
) -> Dict:
    """
    Compute per-class and overall intra-class LPIPS diversity.
    
    Adaptive pairing strategy:
      - If class has <= all_pairs_threshold images: evaluate all unique pairs
      - Otherwise: sample up to max_pairs_per_class random pairs
    
    Args:
        class_to_paths: Dict mapping class names to list of image paths
        lpips_model: Pretrained LPIPS model (custom or standard)
        is_custom: Whether using custom ResNet18 model
        input_mode: 'imagenet' or 'lpips' for transform selection
        device: Device for computation
        image_size: Image resolution for LPIPS
        max_pairs_per_class: Maximum pairs to sample for large classes
        all_pairs_threshold: Threshold for using all pairs vs sampling
    
    Returns:
        Dictionary with per-class and overall statistics
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    tfm = build_lpips_transforms(mode=input_mode, image_size=image_size)

    results = {"per_class": {}, "overall": {"all_distances": []}}

    for cname, paths in class_to_paths.items():
        n = len(paths)
        if n < 2:
            print(f"Skipping class '{cname}': needs at least 2 images")
            continue

        max_pairs = n * (n - 1) // 2
        use_all = (n <= all_pairs_threshold)
        num_pairs = max_pairs if use_all else min(max_pairs_per_class, max_pairs)

        # Generate pairs
        if use_all:
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        else:
            pairs_set = set()
            attempts = 0
            while len(pairs_set) < num_pairs and attempts < num_pairs * 10:
                i, j = random.sample(range(n), 2)
                if i > j:
                    i, j = j, i
                pairs_set.add((i, j))
                attempts += 1
            pairs = list(pairs_set)

        # Compute LPIPS for pairs
        class_dists = []
        for i, j in pairs:
            img1 = tfm(Image.open(paths[i]).convert("RGB")).unsqueeze(0).to(dev)
            img2 = tfm(Image.open(paths[j]).convert("RGB")).unsqueeze(0).to(dev)
            
            if is_custom:
                # Custom model with embed() and pair_distance() methods
                emb1 = lpips_model.embed(img1)
                emb2 = lpips_model.embed(img2)
                d = lpips_model.pair_distance(emb1, emb2).item()
            else:
                # Standard LPIPS model
                d = lpips_model(img1, img2).item()
            
            class_dists.append(float(d))

        if len(class_dists) > 0:
            stats = {
                "mean": float(np.mean(class_dists)),
                "std": float(np.std(class_dists)),
                "min": float(np.min(class_dists)),
                "max": float(np.max(class_dists)),
                "median": float(np.median(class_dists)),
                "n_images": int(n),
                "n_pairs_evaluated": int(len(class_dists)),
                "used_all_pairs": bool(use_all),
            }
            results["per_class"][cname] = stats
            results["overall"]["all_distances"].extend(class_dists)

    # Compute overall statistics
    if len(results["overall"]["all_distances"]) > 0:
        all_d = results["overall"]["all_distances"]
        results["overall"] = {
            "mean": float(np.mean(all_d)),
            "std": float(np.std(all_d)),
            "min": float(np.min(all_d)),
            "max": float(np.max(all_d)),
            "median": float(np.median(all_d)),
            "n_total_pairs": int(len(all_d)),
            "n_classes_evaluated": int(len(results["per_class"])),
        }
    else:
        results["overall"] = {
            "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0,
            "n_total_pairs": 0, "n_classes_evaluated": 0
        }
    
    return results


def group_paths_by_class(root: str) -> Dict[str, List[str]]:
    """Group image paths by class name."""
    ds = ImageFolderDataset(root, transform=None)
    groups: Dict[str, List[str]] = {}
    for path, cidx, cname in ds.samples:
        groups.setdefault(cname, []).append(path)
    return groups


def evaluate_lpips_diversity(
    real_dir: str,
    gen_dir: str,
    output_json: str = "lpips_results.json",
    custom_model_path: Optional[str] = None,
    lpips_backbone: str = "alex",
    image_size: int = 64,
    max_pairs_per_class: int = 200,
    all_pairs_threshold: int = 50,
    device: str = "cuda",
    seed: int = 42,
) -> Dict:
    """
    Main evaluation function for LPIPS intra-class diversity.
    
    Args:
        real_dir: Directory with real images (class subdirectories)
        gen_dir: Directory with generated images (class subdirectories)
        output_json: Path to save JSON results
        custom_model_path: Path to custom ResNet18 model checkpoint (optional)
        lpips_backbone: LPIPS backbone if not using custom model ('alex', 'vgg', or 'squeeze')
        image_size: Image resolution for evaluation
        max_pairs_per_class: Max pairs to sample for large classes
        all_pairs_threshold: Use all pairs if class size <= threshold
        device: Device for computation
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing all results
    """
    set_seed(seed)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")

    # Load model (custom or standard)
    if custom_model_path:
        lpips_model, is_custom, input_mode = load_custom_model(custom_model_path, device=str(dev))
        model_type = "custom_resnet18"
    else:
        lpips_model, is_custom, input_mode = load_standard_lpips(lpips_backbone, device=str(dev))
        model_type = lpips_backbone

    # Index images by class
    print("\nIndexing real images...")
    real_groups = group_paths_by_class(real_dir)
    print(f"Found {len(real_groups)} classes in real directory")
    
    print("Indexing generated images...")
    gen_groups = group_paths_by_class(gen_dir)
    print(f"Found {len(gen_groups)} classes in generated directory")

    # Compute intra-class LPIPS
    print("\nComputing intra-class LPIPS for generated images...")
    intra_gen = compute_intra_class_lpips(
        gen_groups, lpips_model, is_custom, input_mode, device=str(dev),
        image_size=image_size,
        max_pairs_per_class=max_pairs_per_class,
        all_pairs_threshold=all_pairs_threshold,
    )
    
    print("Computing intra-class LPIPS for real images...")
    intra_real = compute_intra_class_lpips(
        real_groups, lpips_model, is_custom, input_mode, device=str(dev),
        image_size=image_size,
        max_pairs_per_class=max_pairs_per_class,
        all_pairs_threshold=all_pairs_threshold,
    )

    # Compute diversity ratio
    gen_mean = intra_gen["overall"]["mean"]
    real_mean = intra_real["overall"]["mean"]
    diversity_ratio = float(gen_mean / real_mean) if real_mean > 0 else 0.0

    # Compile results
    results = {
        "intra_lpips_generated": intra_gen,
        "intra_lpips_real": intra_real,
        "diversity_metrics": {
            "diversity_ratio": diversity_ratio,
            "generated_mean": gen_mean,
            "real_mean": real_mean,
            "interpretation": "Higher ratio indicates more diverse generated images"
        },
        "metadata": {
            "real_dir": real_dir,
            "gen_dir": gen_dir,
            "image_size": int(image_size),
            "max_pairs_per_class": int(max_pairs_per_class),
            "all_pairs_threshold": int(all_pairs_threshold),
            "model_type": model_type,
            "is_custom_model": bool(is_custom),
            "custom_model_path": custom_model_path if custom_model_path else None,
            "seed": int(seed),
            "n_classes": int(len(real_groups)),
        },
    }

    # Save results
    ensure_dir(output_json)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_json}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model: {model_type}")
    print(f"Intra-class LPIPS (generated): {gen_mean:.4f}")
    print(f"Intra-class LPIPS (real):      {real_mean:.4f}")
    print(f"Diversity ratio (gen/real):    {diversity_ratio:.4f}")
    print("\nPer-class results (generated):")
    for cname, stats in intra_gen["per_class"].items():
        print(f"  {cname}: {stats['mean']:.4f} (n={stats['n_images']}, pairs={stats['n_pairs_evaluated']})")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate intra-class LPIPS diversity for generated images."
    )
    parser.add_argument("--real_dir", type=str, required=True,
                        help="Directory with real images (class subdirectories)")
    parser.add_argument("--gen_dir", type=str, required=True,
                        help="Directory with generated images (class subdirectories)")
    parser.add_argument("--output_json", type=str, default="lpips_results.json",
                        help="Output JSON file path")
    parser.add_argument("--custom_model_path", type=str, default=None,
                        help="Path to custom ResNet18 model checkpoint (.pth file)")
    parser.add_argument("--lpips_backbone", type=str, default="alex",
                        choices=["alex", "vgg", "squeeze"],
                        help="LPIPS backbone if not using custom model")
    parser.add_argument("--image_size", type=int, default=64,
                        help="Image resolution for evaluation")
    parser.add_argument("--max_pairs_per_class", type=int, default=200,
                        help="Maximum pairs to sample for large classes")
    parser.add_argument("--all_pairs_threshold", type=int, default=50,
                        help="Use all pairs if class size <= threshold")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for computation (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    evaluate_lpips_diversity(
        real_dir=args.real_dir,
        gen_dir=args.gen_dir,
        output_json=args.output_json,
        custom_model_path=args.custom_model_path,
        lpips_backbone=args.lpips_backbone,
        image_size=args.image_size,
        max_pairs_per_class=args.max_pairs_per_class,
        all_pairs_threshold=args.all_pairs_threshold,
        device=args.device,
        seed=args.seed,
    )
