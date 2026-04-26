"""
train.py — Two-Phase Training Pipeline for PneumoScan

Trains TWO architectures sequentially:
    1. DenseNet121   — Phase 1 (frozen, 10 epochs) + Phase 2 (fine-tune, 10 epochs)
    2. EfficientNetB3 — Phase 1 (frozen, 10 epochs) + Phase 2 (fine-tune, 10 epochs)

Both architectures share:
    • Identical augmented ImageDataGenerator settings
    • Same class weights to counter the NORMAL/PNEUMONIA imbalance
    • Same callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

Outputs
-------
    best_densenet_phase1.keras          — DenseNet121 best Phase 1 checkpoint
    best_densenet_phase2.keras          — DenseNet121 best Phase 2 checkpoint
    best_efficientnet_phase1.keras      — EfficientNetB3 best Phase 1 checkpoint
    best_efficientnet_phase2.keras      — EfficientNetB3 best Phase 2 checkpoint
    phase1_history.pkl                  — DenseNet121 Phase 1 history
    phase2_history.pkl                  — DenseNet121 Phase 2 history
    efficientnet_phase1_history.pkl     — EfficientNetB3 Phase 1 history
    efficientnet_phase2_history.pkl     — EfficientNetB3 Phase 2 history
    training_curves_comparison.png      — 4×2 side-by-side comparison plot
"""

import os
import pickle

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from model import (
    build_phase1_model,
    build_phase2_model,
    build_efficientnet_phase1_model,
    build_efficientnet_phase2_model,
)


# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 10

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_pkl(path: str):
    """Load a pickle file with an informative error on failure."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Required file not found: {path}. Run data_setup.py first."
        )
    except Exception as exc:
        raise RuntimeError(f"Error loading {path}: {exc}") from exc


def _build_dataframe(file_list: list[tuple[str, int]]) -> pd.DataFrame:
    """Convert a list of (filepath, label) tuples to a DataFrame suitable
    for ``flow_from_dataframe``."""
    df = pd.DataFrame(file_list, columns=["filepath", "label"])
    df["label"] = df["label"].astype(str)  # ImageDataGenerator expects str
    return df


def _build_generators(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
):
    """Return (train_generator, val_generator) using the prescribed
    augmentation settings."""

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="filepath",
        y_col="label",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True,
        seed=42,
    )

    val_gen = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col="filepath",
        y_col="label",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )

    return train_gen, val_gen


def _make_callbacks(phase: int, model_name: str = "densenet") -> list:
    """Create a fresh set of Keras callbacks for the given phase and model.

    Parameters
    ----------
    phase : int
        Training phase (1 or 2).
    model_name : str
        Short identifier used in the checkpoint filename
        (e.g. ``'densenet'`` or ``'efficientnet'``).
    """
    checkpoint_path = f"best_{model_name}_phase{phase}.keras"
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_auc",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def _save_history(history, path: str) -> None:
    """Persist a Keras History.history dict as a pickle file."""
    try:
        with open(path, "wb") as f:
            pickle.dump(history.history, f)
        print(f"  ✓ History saved to {path}")
    except IOError as exc:
        print(f"[ERROR] Could not save history to {path}: {exc}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_comparison_curves(
    dn_hist1: dict,
    dn_hist2: dict,
    eff_hist1: dict,
    eff_hist2: dict,
    save_path: str = "training_curves_comparison.png",
) -> None:
    """Create a 4×2 subplot comparing DenseNet121 and EfficientNetB3
    across both training phases (Accuracy and Loss per row).

    Layout
    ------
    Row 0 — Phase 1 Accuracy  : DenseNet121 (left) | EfficientNetB3 (right)
    Row 1 — Phase 1 Loss      : DenseNet121 (left) | EfficientNetB3 (right)
    Row 2 — Phase 2 Accuracy  : DenseNet121 (left) | EfficientNetB3 (right)
    Row 3 — Phase 2 Loss      : DenseNet121 (left) | EfficientNetB3 (right)
    """
    dn_color  = ("#1f77b4", "#aec7e8")   # blue family for DenseNet
    eff_color = ("#ff7f0e", "#ffbb78")   # orange family for EfficientNet

    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    fig.suptitle(
        "PneumoScan — Training Comparison: DenseNet121 vs EfficientNetB3",
        fontsize=15,
        fontweight="bold",
    )

    def _fill_row(row: int, hist: dict, phase: int, model_label: str, colors: tuple) -> None:
        """Populate one accuracy cell and one loss cell for a given row."""
        # Accuracy (column 0 = DenseNet, column 1 = EfficientNet)
        col = 0 if "DenseNet" in model_label else 1
        ax = axes[row, col]
        ax.plot(hist["accuracy"],     label="Train", color=colors[0], linewidth=2)
        ax.plot(hist["val_accuracy"], label="Val",   color=colors[1], linewidth=2, linestyle="--")
        ax.set_title(f"{model_label} — Phase {phase} Accuracy", fontsize=11)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    def _fill_loss_row(row: int, hist: dict, phase: int, model_label: str, colors: tuple) -> None:
        col = 0 if "DenseNet" in model_label else 1
        ax = axes[row, col]
        ax.plot(hist["loss"],     label="Train", color=colors[0], linewidth=2)
        ax.plot(hist["val_loss"], label="Val",   color=colors[1], linewidth=2, linestyle="--")
        ax.set_title(f"{model_label} — Phase {phase} Loss", fontsize=11)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Row 0 — Phase 1 Accuracy
    _fill_row(0, dn_hist1,  phase=1, model_label="DenseNet121",   colors=dn_color)
    _fill_row(0, eff_hist1, phase=1, model_label="EfficientNetB3", colors=eff_color)

    # Row 1 — Phase 1 Loss
    _fill_loss_row(1, dn_hist1,  phase=1, model_label="DenseNet121",   colors=dn_color)
    _fill_loss_row(1, eff_hist1, phase=1, model_label="EfficientNetB3", colors=eff_color)

    # Row 2 — Phase 2 Accuracy
    _fill_row(2, dn_hist2,  phase=2, model_label="DenseNet121",   colors=dn_color)
    _fill_row(2, eff_hist2, phase=2, model_label="EfficientNetB3", colors=eff_color)

    # Row 3 — Phase 2 Loss
    _fill_loss_row(3, dn_hist2,  phase=2, model_label="DenseNet121",   colors=dn_color)
    _fill_loss_row(3, eff_hist2, phase=2, model_label="EfficientNetB3", colors=eff_color)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    try:
        fig.savefig(save_path, dpi=150)
        print(f"  ✓ Comparison training curves saved to {save_path}")
    except IOError as exc:
        print(f"[ERROR] Could not save plot: {exc}")
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("PneumoScan — Training Pipeline")
    print("=" * 60)

    # Load data splits and class weights --------------------------------
    print("\n[LOAD] Loading data splits and class weights …")
    train_files = _load_pkl("train_files.pkl")
    val_files = _load_pkl("val_files.pkl")
    class_weights = _load_pkl("class_weights.pkl")

    print(f"  • Train samples  : {len(train_files)}")
    print(f"  • Val samples    : {len(val_files)}")
    print(f"  • Class weights  : {class_weights}")

    train_df = _build_dataframe(train_files)
    val_df = _build_dataframe(val_files)

    train_gen, val_gen = _build_generators(train_df, val_df)
    print(f"  • Generator class indices: {train_gen.class_indices}")

    # ===================================================================
    # DENSENET121 — PHASE 1 — Feature Extraction
    # ===================================================================
    print("\n" + "=" * 60)
    print("[DenseNet121] PHASE 1 — Feature Extraction (frozen backbone)")
    print("=" * 60)

    dn_model = build_phase1_model()
    callbacks_dn1 = _make_callbacks(phase=1, model_name="densenet")

    dn_history1 = dn_model.fit(
        train_gen,
        epochs=EPOCHS_PHASE1,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks_dn1,
        verbose=1,
    )
    _save_history(dn_history1, "phase1_history.pkl")
    best_dn_auc1 = max(dn_history1.history.get("val_auc", [0]))
    print(f"\n[DenseNet121] Phase 1 complete. Best val_auc: {best_dn_auc1:.4f}")

    # ===================================================================
    # DENSENET121 — PHASE 2 — Fine-Tuning
    # ===================================================================
    print("\n" + "=" * 60)
    print("[DenseNet121] PHASE 2 — Fine-Tuning (last 30 layers)")
    print("=" * 60)

    dn1_ckpt = "best_densenet_phase1.keras"
    try:
        dn_model = tf.keras.models.load_model(dn1_ckpt)
        print(f"  ✓ Loaded best Phase 1 weights from {dn1_ckpt}")
    except Exception as exc:
        print(f"[WARNING] Could not load {dn1_ckpt}: {exc}")
        print("  Continuing with in-memory model.")

    dn_model = build_phase2_model(dn_model)
    callbacks_dn2 = _make_callbacks(phase=2, model_name="densenet")
    train_gen, val_gen = _build_generators(train_df, val_df)  # reset iterators

    dn_history2 = dn_model.fit(
        train_gen,
        epochs=EPOCHS_PHASE2,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks_dn2,
        verbose=1,
    )
    _save_history(dn_history2, "phase2_history.pkl")
    best_dn_auc2 = max(dn_history2.history.get("val_auc", [0]))
    print(f"\n[DenseNet121] Phase 2 complete. Best val_auc: {best_dn_auc2:.4f}")

    # Free DenseNet memory before starting EfficientNet
    del dn_model
    tf.keras.backend.clear_session()

    # Rebuild generators for the EfficientNet runs
    train_gen, val_gen = _build_generators(train_df, val_df)

    # ===================================================================
    # EFFICIENTNETB3 — PHASE 1 — Feature Extraction
    # ===================================================================
    print("\n" + "=" * 60)
    print("[EfficientNetB3] PHASE 1 — Feature Extraction (frozen backbone)")
    print("=" * 60)

    eff_model = build_efficientnet_phase1_model()
    callbacks_eff1 = _make_callbacks(phase=1, model_name="efficientnet")

    eff_history1 = eff_model.fit(
        train_gen,
        epochs=EPOCHS_PHASE1,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks_eff1,
        verbose=1,
    )
    _save_history(eff_history1, "efficientnet_phase1_history.pkl")
    best_eff_auc1 = max(eff_history1.history.get("val_auc", [0]))
    print(f"\n[EfficientNetB3] Phase 1 complete. Best val_auc: {best_eff_auc1:.4f}")

    # ===================================================================
    # EFFICIENTNETB3 — PHASE 2 — Fine-Tuning
    # ===================================================================
    print("\n" + "=" * 60)
    print("[EfficientNetB3] PHASE 2 — Fine-Tuning (last 30 layers)")
    print("=" * 60)

    eff1_ckpt = "best_efficientnet_phase1.keras"
    try:
        eff_model = tf.keras.models.load_model(eff1_ckpt)
        print(f"  ✓ Loaded best Phase 1 weights from {eff1_ckpt}")
    except Exception as exc:
        print(f"[WARNING] Could not load {eff1_ckpt}: {exc}")
        print("  Continuing with in-memory model.")

    eff_model = build_efficientnet_phase2_model(eff_model)
    callbacks_eff2 = _make_callbacks(phase=2, model_name="efficientnet")
    train_gen, val_gen = _build_generators(train_df, val_df)  # reset iterators

    eff_history2 = eff_model.fit(
        train_gen,
        epochs=EPOCHS_PHASE2,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks_eff2,
        verbose=1,
    )
    _save_history(eff_history2, "efficientnet_phase2_history.pkl")
    best_eff_auc2 = max(eff_history2.history.get("val_auc", [0]))
    print(f"\n[EfficientNetB3] Phase 2 complete. Best val_auc: {best_eff_auc2:.4f}")

    # ===================================================================
    # 4×2 Comparison Plot
    # ===================================================================
    print("\n[PLOT] Generating 4×2 comparison training curves …")
    _plot_comparison_curves(
        dn_hist1=dn_history1.history,
        dn_hist2=dn_history2.history,
        eff_hist1=eff_history1.history,
        eff_hist2=eff_history2.history,
    )

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  DenseNet121   Phase 2 best val_auc : {best_dn_auc2:.4f}")
    print(f"  EfficientNetB3 Phase 2 best val_auc: {best_eff_auc2:.4f}")
    print("All training phases complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
