"""
data_setup.py — Dataset Preparation for PneumoScan Chest X-Ray Project

This script merges the original train/ and val/ image directories from the
Kaggle "Chest X-Ray Images (Pneumonia)" dataset, re-splits them into an
80/20 train/validation partition (stratified), computes class weights to
handle the ~1:2.7 class imbalance, and persists the split file-lists and
weights as pickle files for downstream training.

Outputs
-------
    train_files.pkl  : list of (filepath, label) tuples for training
    val_files.pkl    : list of (filepath, label) tuples for validation
    class_weights.pkl: dict {0: weight_normal, 1: weight_pneumonia}
"""

import os
import pickle
import pathlib
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_ROOT = os.path.join("chest_xray")
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VAL_DIR = os.path.join(DATASET_ROOT, "val")
OUTPUT_DIR = "."                       # save artefacts in project root
RANDOM_STATE = 42
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# Label mapping (subfolder name → integer)
LABEL_MAP = {"NORMAL": 0, "PNEUMONIA": 1}


def collect_images(root_dir: str) -> list[tuple[str, int]]:
    """Walk *root_dir* and return a list of (filepath, label) tuples.

    Expects the directory structure:
        root_dir/NORMAL/xxx.jpeg
        root_dir/PNEUMONIA/yyy.jpeg
    """
    samples: list[tuple[str, int]] = []
    root_path = pathlib.Path(root_dir)

    if not root_path.exists():
        raise FileNotFoundError(
            f"Directory not found: {root_path.resolve()}"
        )

    for class_name, label in LABEL_MAP.items():
        class_dir = root_path / class_name
        if not class_dir.exists():
            print(f"[WARNING] Sub-directory missing: {class_dir}")
            continue
        for fpath in class_dir.iterdir():
            if fpath.suffix.lower() in VALID_EXTENSIONS:
                samples.append((str(fpath.resolve()), label))

    return samples


def main() -> None:
    print("=" * 60)
    print("PneumoScan — Data Setup")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Collect all images from train/ AND val/ directories
    # ------------------------------------------------------------------
    print("\n[1/5] Collecting images from train/ and val/ directories …")
    try:
        train_samples = collect_images(TRAIN_DIR)
        val_samples = collect_images(VAL_DIR)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return

    all_samples = train_samples + val_samples
    print(f"  • train/ images : {len(train_samples)}")
    print(f"  • val/ images   : {len(val_samples)}")
    print(f"  • total merged  : {len(all_samples)}")

    if len(all_samples) == 0:
        print("[ERROR] No images found. Check the dataset directory structure.")
        return

    # ------------------------------------------------------------------
    # 2. Stratified 80/20 re-split
    # ------------------------------------------------------------------
    print("\n[2/5] Performing stratified 80/20 split …")
    filepaths = [s[0] for s in all_samples]
    labels = [s[1] for s in all_samples]

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        filepaths, labels,
        test_size=0.20,
        stratify=labels,
        random_state=RANDOM_STATE,
    )

    new_train = list(zip(train_paths, train_labels))
    new_val = list(zip(val_paths, val_labels))

    print(f"  • New train size : {len(new_train)}")
    print(f"  • New val size   : {len(new_val)}")

    # ------------------------------------------------------------------
    # 3. Save split file-lists
    # ------------------------------------------------------------------
    print("\n[3/5] Saving split file-lists …")

    train_pkl_path = os.path.join(OUTPUT_DIR, "train_files.pkl")
    val_pkl_path = os.path.join(OUTPUT_DIR, "val_files.pkl")

    try:
        with open(train_pkl_path, "wb") as f:
            pickle.dump(new_train, f)
        print(f"  ✓ Saved {train_pkl_path}")

        with open(val_pkl_path, "wb") as f:
            pickle.dump(new_val, f)
        print(f"  ✓ Saved {val_pkl_path}")
    except IOError as exc:
        print(f"[ERROR] Failed to write pickle files: {exc}")
        return

    # ------------------------------------------------------------------
    # 4. Class distribution
    # ------------------------------------------------------------------
    print("\n[4/5] Class distribution (new train set):")
    counter = Counter(train_labels)
    inv_label_map = {v: k for k, v in LABEL_MAP.items()}
    for cls_idx in sorted(counter):
        name = inv_label_map[cls_idx]
        print(f"  • {name:>10s} (label {cls_idx}): {counter[cls_idx]:>5d} images")

    total = sum(counter.values())
    ratio = counter[1] / counter[0] if counter[0] > 0 else float("inf")
    print(f"  • Imbalance ratio (PNEUMONIA / NORMAL): {ratio:.2f}")

    # ------------------------------------------------------------------
    # 5. Compute and save class weights
    # ------------------------------------------------------------------
    print("\n[5/5] Computing class weights …")
    classes = np.array(sorted(LABEL_MAP.values()))
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=np.array(train_labels),
    )
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    print(f"  • class_weight dict: {class_weight_dict}")

    weights_pkl_path = os.path.join(OUTPUT_DIR, "class_weights.pkl")
    try:
        with open(weights_pkl_path, "wb") as f:
            pickle.dump(class_weight_dict, f)
        print(f"  ✓ Saved {weights_pkl_path}")
    except IOError as exc:
        print(f"[ERROR] Failed to write class weights: {exc}")
        return

    print("\n" + "=" * 60)
    print("Data setup complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
