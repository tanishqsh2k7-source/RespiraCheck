"""
gradcam.py — Grad-CAM++ Explainability Module for PneumoScan

Native Keras 3 / TensorFlow 2.20 implementation of Grad-CAM++.
No external visualisation libraries required — works directly with the
.keras models produced by train.py.

Functions
---------
    generate_gradcam(model, img_array, class_idx)
        → heatmap array (224, 224) in [0, 1]
    overlay_heatmap(original_img_array, heatmap, alpha)
        → RGB overlay uint8 (224, 224, 3)
    save_gradcam_result(img_path, model, save_path)
        → saves a side-by-side figure to disk
"""

import os
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMG_SIZE = (224, 224)
TARGET_LAYER_NAME = "conv5_block16_2_conv"  # last conv in DenseNet121
LABEL_MAP = {0: "NORMAL", 1: "PNEUMONIA"}

# ---------------------------------------------------------------------------
# Keras version-compatibility shim
# Models saved by a different Keras 3.x minor version may include extra
# serialisation keys (renorm*, quantization_config, synchronized, optional)
# that the currently installed Keras 3.11 does not recognise.
# We strip them at load time so the model can be deserialised cleanly.
# ---------------------------------------------------------------------------
_BATCH_NORM_STRIP = {"renorm", "renorm_clipping", "renorm_momentum", "synchronized"}
_DENSE_STRIP = {"quantization_config"}
_INPUT_LAYER_STRIP = {"optional"}


def safe_load_model(path: str) -> tf.keras.Model:
    """Load a .keras model with cross-version compatibility fixes."""
    _orig_bn_from_config = tf.keras.layers.BatchNormalization.from_config
    _orig_dense_from_config = tf.keras.layers.Dense.from_config
    _orig_input_from_config = tf.keras.layers.InputLayer.from_config

    @classmethod  # type: ignore[misc]
    def _patched_bn_from_config(cls, config):
        for key in _BATCH_NORM_STRIP:
            config.pop(key, None)
        return _orig_bn_from_config.__func__(cls, config)

    @classmethod  # type: ignore[misc]
    def _patched_dense_from_config(cls, config):
        for key in _DENSE_STRIP:
            config.pop(key, None)
        return _orig_dense_from_config.__func__(cls, config)

    @classmethod  # type: ignore[misc]
    def _patched_input_from_config(cls, config):
        for key in _INPUT_LAYER_STRIP:
            config.pop(key, None)
        # Keras 3.11 expects 'shape' not 'batch_shape'
        if "batch_shape" in config and "shape" not in config:
            batch_shape = config.pop("batch_shape")
            config["shape"] = batch_shape[1:] if batch_shape else None
        return _orig_input_from_config.__func__(cls, config)

    # Monkey-patch
    tf.keras.layers.BatchNormalization.from_config = _patched_bn_from_config
    tf.keras.layers.Dense.from_config = _patched_dense_from_config
    tf.keras.layers.InputLayer.from_config = _patched_input_from_config

    try:
        model = tf.keras.models.load_model(path)
    finally:
        # Restore originals
        tf.keras.layers.BatchNormalization.from_config = _orig_bn_from_config
        tf.keras.layers.Dense.from_config = _orig_dense_from_config
        tf.keras.layers.InputLayer.from_config = _orig_input_from_config

    return model


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_last_conv_layer(model: tf.keras.Model) -> str:
    """Return the name of the last Conv2D layer in the model.

    Tries ``TARGET_LAYER_NAME`` first; falls back to a reverse scan.
    """
    layer_names = [l.name for l in model.layers]

    if TARGET_LAYER_NAME in layer_names:
        return TARGET_LAYER_NAME

    # Fallback: find the last Conv2D layer
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            print(f"[gradcam] Fallback target layer: {layer.name}")
            return layer.name

    raise ValueError("No Conv2D layer found in the model.")


def _preprocess_image(img_path: str) -> np.ndarray:
    """Load and preprocess an image for inference.

    Returns
    -------
    np.ndarray
        Shape (1, 224, 224, 3), normalised to [0, 1].
    """
    try:
        img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not load image at {img_path}: {exc}"
        ) from exc

    img_array = tf.keras.utils.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_gradcam(
    model: tf.keras.Model,
    img_array: np.ndarray,
    class_idx: int = 0,
) -> np.ndarray:
    """Generate a Grad-CAM++ heatmap for a given image.

    Uses native TensorFlow GradientTape — no external libraries needed.

    Parameters
    ----------
    model : tf.keras.Model
        Trained binary classifier.
    img_array : np.ndarray
        Preprocessed image of shape ``(1, 224, 224, 3)`` in [0, 1].
    class_idx : int
        Target class index (0 or 1).

    Returns
    -------
    np.ndarray
        Heatmap of shape ``(224, 224)`` with float values in [0, 1].
    """
    target_layer_name = _find_last_conv_layer(model)

    # Build a sub-model that outputs both the conv layer activations and
    # the final prediction.
    conv_layer = model.get_layer(target_layer_name)
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[conv_layer.output, model.output],
    )

    # Forward pass under nested GradientTapes for higher-order derivatives
    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape3:
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape1:
                tape1.watch(img_tensor)
                conv_output, predictions = grad_model(img_tensor, training=False)

                # For binary sigmoid: score the predicted class
                if class_idx == 1:
                    score = predictions[:, 0]
                else:
                    score = 1.0 - predictions[:, 0]

            # First derivative
            grads_1 = tape1.gradient(score, conv_output)

        # Second derivative
        if grads_1 is not None:
            grads_2 = tape2.gradient(grads_1, conv_output)
        else:
            grads_2 = None

    # Third derivative
    if grads_2 is not None:
        grads_3 = tape3.gradient(grads_2, conv_output)
    else:
        grads_3 = None

    # Handle None gradients
    if grads_1 is None:
        print("[gradcam] Warning: first-order gradients are None, returning blank heatmap.")
        return np.zeros((IMG_SIZE[0], IMG_SIZE[1]), dtype=np.float32)

    conv_output_val = conv_output[0]  # (H, W, C)
    grads_1_val = grads_1[0]

    # Decide: Grad-CAM++ (if higher-order grads available) or standard Grad-CAM
    use_plus_plus = (grads_2 is not None)

    if use_plus_plus:
        grads_2_val = grads_2[0]
        grads_3_val = grads_3[0] if grads_3 is not None else tf.zeros_like(grads_2[0])

        # Grad-CAM++ alpha weights
        global_sum = tf.reduce_sum(
            tf.reshape(conv_output_val, [-1, conv_output_val.shape[-1]]) *
            tf.reshape(grads_3_val, [-1, grads_3_val.shape[-1]]),
            axis=0,
        )  # shape: (C,)

        denom = 2.0 * grads_2_val + global_sum[tf.newaxis, tf.newaxis, :] + 1e-10
        alpha = grads_2_val / denom  # (H, W, C)
        alpha = alpha * tf.nn.relu(grads_1_val)

        weights = tf.reduce_sum(alpha, axis=(0, 1))  # (C,)
    else:
        # Standard Grad-CAM: global average pooling of gradients
        print("[gradcam] Higher-order gradients unavailable, using standard Grad-CAM.")
        weights = tf.reduce_mean(grads_1_val, axis=(0, 1))  # (C,)

    # Weighted combination of conv feature maps
    heatmap = tf.reduce_sum(conv_output_val * weights[tf.newaxis, tf.newaxis, :], axis=-1)

    # ReLU — we only care about positive influence
    heatmap = tf.nn.relu(heatmap)

    # Resize to input image size
    heatmap = tf.image.resize(
        heatmap[tf.newaxis, ..., tf.newaxis],
        IMG_SIZE,
        method="bilinear",
    )[0, :, :, 0].numpy()

    # Normalise to [0, 1]
    hmin, hmax = heatmap.min(), heatmap.max()
    if hmax - hmin > 1e-8:
        heatmap = (heatmap - hmin) / (hmax - hmin)
    else:
        heatmap = np.zeros_like(heatmap)

    return heatmap.astype(np.float32)


def overlay_heatmap(
    original_img_array: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay a heatmap on the original image using JET colourmap.

    Parameters
    ----------
    original_img_array : np.ndarray
        RGB image of shape ``(224, 224, 3)`` with float values in [0, 1].
    heatmap : np.ndarray
        Heatmap of shape ``(224, 224)`` with float values in [0, 1].
    alpha : float
        Blending factor for the heatmap overlay.

    Returns
    -------
    np.ndarray
        Blended RGB image of shape ``(224, 224, 3)`` as uint8.
    """
    # Convert heatmap to uint8 and apply JET colourmap
    heatmap_uint8 = np.uint8(255 * heatmap)
    jet_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    jet_heatmap = cv2.cvtColor(jet_heatmap, cv2.COLOR_BGR2RGB)  # BGR → RGB

    # Convert original to uint8
    original_uint8 = np.uint8(255 * np.clip(original_img_array, 0, 1))

    # Alpha blending
    overlay = cv2.addWeighted(original_uint8, 1 - alpha, jet_heatmap, alpha, 0)
    return overlay


def save_gradcam_result(
    img_path: str,
    model: tf.keras.Model,
    save_path: str,
) -> None:
    """Load an image, run Grad-CAM++, and save a side-by-side figure.

    The figure shows [Original X-ray | Grad-CAM++ Overlay] with the
    predicted label and confidence score as the suptitle.

    Parameters
    ----------
    img_path : str
        Path to the input chest X-ray image.
    model : tf.keras.Model
        Trained PneumoScan model.
    save_path : str
        Output path for the saved figure.
    """
    # Preprocess
    img_array = _preprocess_image(img_path)         # (1, 224, 224, 3)
    original = img_array[0]                          # (224, 224, 3)

    # Predict
    pred_prob = float(model.predict(img_array, verbose=0)[0, 0])
    pred_class = 1 if pred_prob > 0.5 else 0
    confidence = pred_prob if pred_class == 1 else 1 - pred_prob
    label = LABEL_MAP[pred_class]

    # Generate heatmap
    heatmap = generate_gradcam(model, img_array, class_idx=pred_class)
    overlay = overlay_heatmap(original, heatmap)

    # Plot side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(
        f"Prediction: {label}  |  Confidence: {confidence:.1%}",
        fontsize=14,
        fontweight="bold",
    )

    axes[0].imshow(original)
    axes[0].set_title("Original X-Ray")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("Grad-CAM++ Overlay")
    axes[1].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    try:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [OK] Grad-CAM result saved to {save_path}")
    except IOError as exc:
        print(f"[ERROR] Could not save Grad-CAM figure: {exc}")
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Grad-CAM++ overlay")
    parser.add_argument("--image", required=True, help="Path to a chest X-ray image")
    parser.add_argument(
        "--model",
        default="best_densenet_phase2.keras",
        help="Path to the trained model",
    )
    parser.add_argument(
        "--output",
        default="gradcam_output.png",
        help="Path to save the output figure",
    )
    args = parser.parse_args()

    print("[gradcam] Loading model …")
    loaded_model = safe_load_model(args.model)
    print("[gradcam] Generating Grad-CAM++ …")
    save_gradcam_result(args.image, loaded_model, args.output)
    print("[gradcam] Done.")
