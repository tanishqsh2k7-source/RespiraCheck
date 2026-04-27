import os
from dataclasses import dataclass
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

DENSENET_PATH    = os.environ.get("DENSENET_PATH",    "best_densenet_phase2.keras")
EFFICIENTNET_PATH = os.environ.get("EFFICIENTNET_PATH", "best_efficientnet_phase2.keras")
TEST_DIR    = os.path.join("chest_xray", "test")
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
TARGET_NAMES = ["NORMAL", "PNEUMONIA"]

@dataclass
class ModelResult:
    name: str
    path: str
    test_loss: float
    test_acc: float
    roc_auc: float
    ap_score: float
    pneumonia_recall: float
    pneumonia_precision: float
    pneumonia_f1: float
    true_classes: np.ndarray
    pred_classes: np.ndarray
    pred_probs: np.ndarray


def _load_model(path: str) -> tf.keras.Model:
    #Load a .keras model with informative error handling.
    print(f"[LOAD] Loading model from {path} …")
    try:
        model = tf.keras.models.load_model(path)
        print("  ✓ Model loaded successfully.")
        return model
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model from {path}: {exc}\n"
            "Make sure you have run train.py first."
        ) from exc

def _build_test_generator(test_dir: str):
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    gen = datagen.flow_from_directory(
        directory=test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )
    return gen

def _evaluate_model(model: tf.keras.Model, model_name: str, model_path: str) -> ModelResult:
    # Build a fresh generator so the cursor is at position 0
    test_gen = _build_test_generator(TEST_DIR)
    print(f"\n  • Test samples : {test_gen.samples}")
    print(f"  • Class indices: {test_gen.class_indices}")

    # Keras evaluate
    print(f"\n[EVAL:{model_name}] Running model.evaluate() …")
    test_loss, test_acc, _ = model.evaluate(test_gen, verbose=1)

    # Predictions
    print(f"[PRED:{model_name}] Generating predictions …")
    test_gen.reset()
    pred_probs  = model.predict(test_gen, verbose=1).flatten()
    pred_classes = (pred_probs > 0.5).astype(int)
    true_classes = test_gen.classes

    # Per-class metrics (PNEUMONIA = label 1)
    pneu_recall    = recall_score(true_classes, pred_classes, pos_label=1)
    pneu_precision = precision_score(true_classes, pred_classes, pos_label=1, zero_division=0)
    pneu_f1        = f1_score(true_classes, pred_classes, pos_label=1, zero_division=0)

    # ROC-AUC
    fpr, tpr, _ = roc_curve(true_classes, pred_probs)
    roc_auc_val  = auc(fpr, tpr)

    # Average Precision
    ap = average_precision_score(true_classes, pred_probs)

    print(f"  • Test Accuracy    : {test_acc:.4f}")
    print(f"  • ROC-AUC          : {roc_auc_val:.4f}")
    print(f"  • Pneumonia Recall : {pneu_recall:.4f}")

    return ModelResult(
        name=model_name,
        path=model_path,
        test_loss=test_loss,
        test_acc=test_acc,
        roc_auc=roc_auc_val,
        ap_score=ap,
        pneumonia_recall=pneu_recall,
        pneumonia_precision=pneu_precision,
        pneumonia_f1=pneu_f1,
        true_classes=true_classes,
        pred_classes=pred_classes,
        pred_probs=pred_probs,
    )

def _plot_confusion_matrix(result: ModelResult, save_path: str) -> None:
    cm = confusion_matrix(result.true_classes, result.pred_classes)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=TARGET_NAMES,
        yticklabels=TARGET_NAMES,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix — {result.name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Confusion matrix saved to {save_path}")

def _plot_roc_comparison(dn: ModelResult, eff: ModelResult, save_path: str) -> None:
    fpr_dn,  tpr_dn,  _ = roc_curve(dn.true_classes,  dn.pred_probs)
    fpr_eff, tpr_eff, _ = roc_curve(eff.true_classes, eff.pred_probs)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr_dn,  tpr_dn,  color="#1f77b4", linewidth=2,
            label=f"DenseNet121    AUC = {dn.roc_auc:.4f}")
    ax.plot(fpr_eff, tpr_eff, color="#ff7f0e", linewidth=2,
            label=f"EfficientNetB3 AUC = {eff.roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ ROC comparison saved to {save_path}")

def _plot_pr_comparison(dn: ModelResult, eff: ModelResult, save_path: str) -> None:
    prec_dn,  rec_dn,  _ = precision_recall_curve(dn.true_classes,  dn.pred_probs)
    prec_eff, rec_eff, _ = precision_recall_curve(eff.true_classes, eff.pred_probs)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rec_dn,  prec_dn,  color="#1f77b4", linewidth=2,
            label=f"DenseNet121    AP = {dn.ap_score:.4f}")
    ax.plot(rec_eff, prec_eff, color="#ff7f0e", linewidth=2,
            label=f"EfficientNetB3 AP = {eff.ap_score:.4f}")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision–Recall Curve Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ PR comparison saved to {save_path}")

def _comparison_table(dn: ModelResult, eff: ModelResult) -> str:
    col_w = 15

    header = (
        f"{'Metric':<30} | {'DenseNet121':>{col_w}} | {'EfficientNetB3':>{col_w}}\n"
        + "-" * (30 + col_w * 2 + 6) + "\n"
    )
    rows = [
        ("Test Accuracy",        f"{dn.test_acc:.4f}",         f"{eff.test_acc:.4f}"),
        ("Pneumonia Recall",     f"{dn.pneumonia_recall:.4f}",  f"{eff.pneumonia_recall:.4f}"),
        ("Pneumonia Precision",  f"{dn.pneumonia_precision:.4f}", f"{eff.pneumonia_precision:.4f}"),
        ("F1-Score (Pneumonia)", f"{dn.pneumonia_f1:.4f}",     f"{eff.pneumonia_f1:.4f}"),
        ("AUC-ROC",              f"{dn.roc_auc:.4f}",          f"{eff.roc_auc:.4f}"),
        ("Avg Precision (AP)",   f"{dn.ap_score:.4f}",         f"{eff.ap_score:.4f}"),
        ("Test Loss",            f"{dn.test_loss:.4f}",         f"{eff.test_loss:.4f}"),
    ]

    body = "".join(
        f"{label:<30} | {dn_val:>{col_w}} | {eff_val:>{col_w}}\n"
        for label, dn_val, eff_val in rows
    )
    return header + body

def _per_model_section(result: ModelResult) -> str:
    cls_report = classification_report(
        result.true_classes, result.pred_classes, target_names=TARGET_NAMES
    )
    return (
        f"\n{'=' * 50}\n"
        f"Model: {result.name}  ({result.path})\n"
        f"{'=' * 50}\n"
        f"Test Loss     : {result.test_loss:.4f}\n"
        f"Test Accuracy : {result.test_acc:.4f}\n"
        f"ROC-AUC       : {result.roc_auc:.4f}\n"
        f"Avg Precision : {result.ap_score:.4f}\n\n"
        f"Classification Report\n"
        f"{'-' * 50}\n"
        f"{cls_report}\n"
        f"Confusion Matrix\n"
        f"{'-' * 50}\n"
        f"{confusion_matrix(result.true_classes, result.pred_classes)}\n"
    )

def main() -> None:
    print("\nPneumoScan — Dual-Model Evaluation & Comparison :\n")
    dn_model  = _load_model(DENSENET_PATH)
    eff_model = _load_model(EFFICIENTNET_PATH)

    # Evaluate each model
    print("\n")
    print("Evaluating DenseNet121 …\n")
    dn_result = _evaluate_model(dn_model, "DenseNet121", DENSENET_PATH)
    print("\n")
    print("Evaluating EfficientNetB3 …\n")
    eff_result = _evaluate_model(eff_model, "EfficientNetB3", EFFICIENTNET_PATH)

    # Per-model confusion matrices
    print("\n[PLOT] Saving per-model confusion matrices …")
    _plot_confusion_matrix(dn_result,  "confusion_matrix_densenet.png")
    _plot_confusion_matrix(eff_result, "confusion_matrix_efficientnet.png")

    # Combined comparison plots
    print("\n[PLOT] Saving ROC and PR comparison curves …")
    _plot_roc_comparison(dn_result, eff_result, "roc_curve_comparison.png")
    _plot_pr_comparison(dn_result,  eff_result, "pr_curve_comparison.png")

    # Comparison table
    table = _comparison_table(dn_result, eff_result)
    print("\n")
    print("Side-by-Side Comparison\n")
    print(table)

    # Determine best model (by ROC-AUC)
    if dn_result.roc_auc >= eff_result.roc_auc:
        best_result = dn_result
    else:
        best_result = eff_result

    verdict = (
        f"Best Model: {best_result.name} based on AUC-ROC score "
        f"({best_result.roc_auc:.4f})"
    )
    print(verdict)

    try:
        with open("best_model_path.txt", "w", encoding="utf-8") as f:
            f.write(best_result.path)
        print(f"  ✓ Best model path written to best_model_path.txt")
    except IOError as exc:
        print(f"[ERROR] Could not write best_model_path.txt: {exc}")

    for res in (dn_result, eff_result):
        acc_met    = "✓" if res.test_acc          > 0.80 else "✗"
        recall_met = "✓" if res.pneumonia_recall  > 0.93 else "✗"
        auc_met    = "✓" if res.roc_auc           > 0.92 else "✗"
        print(
            f"[{res.name}] Target met: Accuracy>80%: {acc_met} ({res.test_acc:.2%}) | "
            f"Pneumonia Recall>93%: {recall_met} ({res.pneumonia_recall:.2%}) | "
            f"AUC>0.92: {auc_met} ({res.roc_auc:.4f})"
        )
    # Save full evaluation report
    report_path = "evaluation_report.txt"
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("Respiracheck — Dual-Model Evaluation Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(_per_model_section(dn_result))
            f.write(_per_model_section(eff_result))
            f.write("\n\nSide-by-Side Comparison\n")
            f.write("=" * 60 + "\n")
            f.write(table + "\n")
            f.write(verdict + "\n")

        print(f"\n  ✓ Full report saved to {report_path}")
    except IOError as exc:
        print(f"[ERROR] Could not save report: {exc}")

    print("\n")
    print("Evaluation complete.")

if __name__ == "__main__":
    main()