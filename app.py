"""
app.py — Flask REST API for PneumoScan

Serves the best-performing chest X-ray classifier (DenseNet121 or
EfficientNetB3) with Grad-CAM++ explainability.  The active model is
determined automatically at startup by reading ``best_model_path.txt``,
which is written by ``evaluate.py`` after the two-model comparison.

Model resolution order
----------------------
    1. ``best_model_path.txt``  (written by evaluate.py — preferred)
    2. ``MODEL_PATH`` environment variable
    3. Hard-coded fallback: ``best_densenet_phase2.keras``

Endpoints
---------
    POST /predict   — Upload a chest X-ray, receive prediction + Grad-CAM overlay
    GET  /health    — Liveness probe
"""

import base64
import io
import os

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image

from gradcam import generate_gradcam, overlay_heatmap

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BEST_MODEL_PTR   = "best_model_path.txt"          # written by evaluate.py
FALLBACK_PATH    = os.environ.get("MODEL_PATH", "best_densenet_phase2.keras")
IMG_SIZE         = (224, 224)
MAX_CONTENT_LENGTH = 5 * 1024 * 1024              # 5 MB
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
LABEL_MAP        = {0: "NORMAL", 1: "PNEUMONIA"}


def _resolve_model_path() -> str:
    """Determine which model file to load.

    Resolution order:
        1. ``best_model_path.txt`` written by evaluate.py
        2. ``MODEL_PATH`` environment variable
        3. Hard-coded fallback (best_densenet_phase2.keras)

    Returns
    -------
    str
        Absolute-or-relative path to the .keras model file.
    """
    if os.path.isfile(BEST_MODEL_PTR):
        try:
            with open(BEST_MODEL_PTR, "r", encoding="utf-8") as f:
                path = f.read().strip()
            if path and os.path.isfile(path):
                print(f"[app] Model resolved from {BEST_MODEL_PTR}: {path}")
                return path
            else:
                print(
                    f"[WARNING] {BEST_MODEL_PTR} contains '{path}' which does "
                    "not exist. Falling back."
                )
        except IOError as exc:
            print(f"[WARNING] Could not read {BEST_MODEL_PTR}: {exc}. Falling back.")

    print(f"[app] Using fallback model path: {FALLBACK_PATH}")
    return FALLBACK_PATH

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
CORS(app)  # allow cross-origin requests (React dev server on port 3000)

# Load model once at startup
model      = None
model_path = None   # set by _load_model(); exposed on /health


def _load_model() -> None:
    """Resolve and load the best model from disk (called once at startup)."""
    global model, model_path
    model_path = _resolve_model_path()
    try:
        print(f"[app] Loading model from {model_path} …")
        model = tf.keras.models.load_model(model_path)
        print("[app] Model loaded successfully.")
    except Exception as exc:
        print(f"[ERROR] Failed to load model from {model_path}: {exc}")
        model = None


def _allowed_file(filename: str) -> bool:
    """Return True if the filename has an allowed image extension."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def _preprocess_image(file_storage) -> np.ndarray:
    """Read an uploaded file into a preprocessed numpy array.

    Returns
    -------
    np.ndarray
        Shape (1, 224, 224, 3) normalised to [0, 1].
    """
    img = Image.open(file_storage.stream).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


def _encode_overlay_to_base64(overlay: np.ndarray) -> str:
    """Encode an RGB uint8 overlay image to a base64 PNG string."""
    img = Image.fromarray(overlay)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """Liveness / readiness probe."""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": model_path,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """Accept a chest X-ray image and return prediction + Grad-CAM overlay."""
    # --- Validate model availability ---
    if model is None:
        return jsonify({"error": "Model not loaded. Check server logs."}), 503

    # --- Validate uploaded file ---
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not _allowed_file(file.filename):
        return jsonify({
            "error": (
                f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        }), 400

    # --- Preprocess ---
    try:
        img_array = _preprocess_image(file)  # (1, 224, 224, 3)
    except Exception as exc:
        return jsonify({"error": f"Image preprocessing failed: {exc}"}), 400

    # --- Predict ---
    try:
        pred_prob = float(model.predict(img_array, verbose=0)[0, 0])
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    pred_class = 1 if pred_prob > 0.5 else 0
    confidence = pred_prob if pred_class == 1 else 1 - pred_prob
    label = LABEL_MAP[pred_class]

    # --- Grad-CAM++ ---
    try:
        heatmap = generate_gradcam(model, img_array, class_idx=pred_class)
        overlay = overlay_heatmap(img_array[0], heatmap, alpha=0.4)
        gradcam_b64 = _encode_overlay_to_base64(overlay)
    except Exception as exc:
        print(f"[WARNING] Grad-CAM generation failed: {exc}")
        gradcam_b64 = ""  # non-fatal — still return the prediction

    return jsonify({
        "prediction": label,
        "confidence": round(confidence, 4),
        "gradcam_image": gradcam_b64,
    })


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        "error": f"File too large. Maximum allowed size is "
                 f"{MAX_CONTENT_LENGTH // (1024 * 1024)} MB."
    }), 413


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _load_model()
    print("[app] Starting Flask server on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
