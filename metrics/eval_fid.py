import os
import argparse
import numpy as np
from pathlib import Path
from glob import glob
from tqdm import tqdm

from cleanfid import fid as clean_fid


def detect_directory_structure(directory):
    """
    Detect if directory has subdirectories (multi-class) or contains images directly (single-class).
    
    Returns:
        tuple: (is_multi_class, class_names_or_none)
    """
    path = Path(directory)
    
    # Check for subdirectories
    subdirs = [d for d in path.iterdir() if d.is_dir()]
    
    # Check for image files directly in the directory
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    direct_images = []
    for ext in image_extensions:
        direct_images.extend(list(path.glob(ext)))
        direct_images.extend(list(path.glob(ext.upper())))
    
    if subdirs and not direct_images:
        # Multi-class: has subdirectories but no direct images
        class_names = sorted([d.name for d in subdirs])
        return True, class_names
    elif direct_images and not subdirs:
        # Single-class: has direct images but no subdirectories
        return False, None
    elif subdirs and direct_images:
        print(f"Warning: Directory '{directory}' contains both subdirectories and direct images.")
        print("Treating as multi-class (ignoring direct images).")
        class_names = sorted([d.name for d in subdirs])
        return True, class_names
    else:
        raise ValueError(f"Directory '{directory}' appears to be empty or contains no valid images.")


def compute_classwise_fid(real_root, fake_root, is_multi_class, class_names, dataset_res=64):
    """
    Compute FID for both multi-class and single-class scenarios.
    
    Args:
        real_root: Path to directory with real images
        fake_root: Path to directory with generated images
        is_multi_class: Whether directories have class subdirectories
        class_names: List of class names (None for single-class)
        dataset_res: Resolution for FID computation (default: 64)
    
    Returns:
        float or dict: Single FID score for single-class, dict of scores for multi-class
    """
    if not is_multi_class:
        # Single-class evaluation
        if not os.path.exists(real_root) or not os.path.exists(fake_root):
            print("Warning: One or both directories do not exist.")
            return None

        try:
            print("Computing FID...")
            score = clean_fid.compute_fid(real_root, fake_root, mode='clean', dataset_res=dataset_res)
            return score
        except Exception as e:
            print(f"Error computing FID: {e}")
            return None
    
    # Multi-class evaluation
    fid_scores = {}
    for cls in tqdm(class_names, desc="Computing FID (clean-fid)"):
        real_cls_path = os.path.join(real_root, cls)
        fake_cls_path = os.path.join(fake_root, cls)

        if not os.path.exists(real_cls_path) or not os.path.exists(fake_cls_path):
            print(f"Warning: Directory for class '{cls}' does not exist in one or both roots.")
            fid_scores[cls] = None
            continue
        try:
            score = clean_fid.compute_fid(real_cls_path, fake_cls_path, mode='clean', dataset_res=dataset_res)
            fid_scores[cls] = score
        except Exception as e:
            print(f"Warning: Skipped class '{cls}' due to error: {e}")
            fid_scores[cls] = None

    return fid_scores


def save_results(output_path, fid_scores, is_multi_class, class_names=None):
    """Save FID results to file, handling both multi-class and single-class formats."""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        if is_multi_class:
            # Multi-class output format
            f.write("Class-wise FID (clean-fid):\n")
            f.write("=" * 50 + "\n")
            for cls in class_names:
                if fid_scores[cls] is not None:
                    f.write(f"{cls}: {fid_scores[cls]:.4f}\n")
                else:
                    f.write(f"{cls}: N/A\n")
            
            # Calculate and write average
            valid_fids = [v for v in fid_scores.values() if v is not None]
            if valid_fids:
                f.write("\n" + "=" * 50 + "\n")
                f.write(f"Average FID: {np.mean(valid_fids):.4f}\n")
        else:
            # Single-class output format
            f.write("FID Score (clean-fid):\n")
            f.write("=" * 50 + "\n")
            if fid_scores is not None:
                f.write(f"FID: {fid_scores:.4f}\n")
            else:
                f.write("FID: N/A\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate generated images using FID metric. "
                    "Supports both multi-class (subdirectories) and single-class (flat directory) structures."
    )
    parser.add_argument('--real_dir', type=str, required=True, 
                        help="Directory with real images.")
    parser.add_argument('--fake_dir', type=str, required=True, 
                        help="Directory with generated images.")
    parser.add_argument('--output', type=str, required=True, 
                        help="File to save results (e.g., fid_results.txt).")
    parser.add_argument('--dataset_res', type=int, default=64, 
                        help="Resolution for FID computation (default: 64).")
    args = parser.parse_args()

    # Detect directory structures
    print("Detecting directory structures...")
    real_is_multi_class, real_class_names = detect_directory_structure(args.real_dir)
    fake_is_multi_class, fake_class_names = detect_directory_structure(args.fake_dir)

    print(f"Real directory: {'Multi-class' if real_is_multi_class else 'Single-class'}")
    print(f"Fake directory: {'Multi-class' if fake_is_multi_class else 'Single-class'}")

    # Check consistency
    if real_is_multi_class != fake_is_multi_class:
        raise ValueError("Real and fake directories must have the same structure (both multi-class or both single-class).")
    
    if real_is_multi_class and set(real_class_names) != set(fake_class_names):
        print("Warning: Class names don't match between real and fake directories.")
        print(f"Real classes: {real_class_names}")
        print(f"Fake classes: {fake_class_names}")
        # Use intersection of class names
        class_names = sorted(list(set(real_class_names) & set(fake_class_names)))
        print(f"Using common classes: {class_names}")
    else:
        class_names = real_class_names if real_is_multi_class else None

    is_multi_class = real_is_multi_class

    # Compute FID
    fid_scores = compute_classwise_fid(
        args.real_dir, 
        args.fake_dir, 
        is_multi_class, 
        class_names,
        dataset_res=args.dataset_res
    )
    
    # Save results
    print("Saving results...")
    save_results(args.output, fid_scores, is_multi_class, class_names)
    print(f"Results saved to {args.output}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    if is_multi_class:
        print(f"Evaluated {len(class_names)} classes")
        if fid_scores:
            valid_fids = [v for v in fid_scores.values() if v is not None]
            if valid_fids:
                print(f"Average FID: {np.mean(valid_fids):.4f}")
                print("\nPer-class FID scores:")
                for cls in class_names:
                    if fid_scores[cls] is not None:
                        print(f"  {cls}: {fid_scores[cls]:.4f}")
    else:
        print("Single-class evaluation")
        if fid_scores is not None:
            print(f"FID: {fid_scores:.4f}")
        else:
            print("FID: N/A")