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

from gradcam import generate_gradcam, overlay_heatmap, safe_load_model
from dotenv import load_dotenv
load_dotenv()  

try:
    from google import genai

    _GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    if _GEMINI_API_KEY:
        _gemini_client = genai.Client(api_key=_GEMINI_API_KEY)
        print("[app] Gemini 2.5 Flash configured successfully (google-genai).")
    else:
        _gemini_client = None
        print("[WARNING] GEMINI_API_KEY not set. /chat endpoint will be unavailable.")
except ImportError:
    _gemini_client = None
    print("[WARNING] google-genai not installed. /chat endpoint disabled.")

BEST_MODEL_PTR   = "best_model_path.txt"          
FALLBACK_PATH    = os.environ.get("MODEL_PATH", "best_densenet_phase2.keras")
IMG_SIZE         = (224, 224)
MAX_CONTENT_LENGTH = 5 * 1024 * 1024              
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
LABEL_MAP        = {0: "NORMAL", 1: "PNEUMONIA"}

SYSTEM_PROMPT = """You are PneumoScan AI Assistant, an intelligent medical AI helper integrated into a chest X-ray pneumonia detection system. Your role is to:

1. **Explain predictions**: When given prediction context (the model's classification and confidence), explain what the result means in clear, non-technical language.

2. **Educate about pneumonia**: Provide accurate, helpful information about pneumonia — types (bacterial, viral, fungal), symptoms, risk factors, treatment approaches, and recovery.

3. **Guide next steps**: If pneumonia is detected, suggest appropriate next steps (seeing a doctor, getting additional tests, etc.).

4. **Explain the AI model**: If asked, explain how the deep learning model works (DenseNet/EfficientNet transfer learning, Grad-CAM++ heatmaps) in accessible terms.

IMPORTANT RULES:
- Always emphasize that you are an AI screening tool, NOT a substitute for professional medical diagnosis.
- Never provide specific treatment prescriptions or dosages.
- Always recommend consulting a healthcare professional for definitive diagnosis.
- Be empathetic and reassuring while remaining factual.
- Keep responses concise (2-4 paragraphs max) unless the user asks for detailed explanations.
- Use markdown formatting (bold, bullet points) for readability.
- If the prediction is NORMAL, reassure the user but still recommend regular checkups.
- If the prediction is PNEUMONIA, be calm and informative, not alarming."""


def _resolve_model_path() -> str:
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

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
CORS(app) 

model      = None
model_path = None   

def _load_model() -> None:
    global model, model_path
    model_path = _resolve_model_path()
    try:
        print(f"[app] Loading model from {model_path} …")
        model = safe_load_model(model_path)
        print("[app] Model loaded successfully.")
    except Exception as exc:
        print(f"[ERROR] Failed to load model from {model_path}: {exc}")
        model = None

def _allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )

def _preprocess_image(file_storage) -> np.ndarray:
    img = Image.open(file_storage.stream).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def _encode_overlay_to_base64(overlay: np.ndarray) -> str:
    img = Image.fromarray(overlay)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


@app.route("/health", methods=["GET"])
def health():
    """Liveness / readiness probe."""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": model_path,
        "gemini_available": _gemini_client is not None,
    })

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Check server logs."}), 503

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

    try:
        img_array = _preprocess_image(file)  
    except Exception as exc:
        return jsonify({"error": f"Image preprocessing failed: {exc}"}), 400

    # Predict
    try:
        pred_prob = float(model.predict(img_array, verbose=0)[0, 0])
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    pred_class = 1 if pred_prob > 0.5 else 0
    confidence = pred_prob if pred_class == 1 else 1 - pred_prob
    label = LABEL_MAP[pred_class]

    # Grad-CAM++
    try:
        heatmap = generate_gradcam(model, img_array, class_idx=pred_class)
        overlay = overlay_heatmap(img_array[0], heatmap, alpha=0.4)
        gradcam_b64 = _encode_overlay_to_base64(overlay)
    except Exception as exc:
        print(f"[WARNING] Grad-CAM generation failed: {exc}")
        gradcam_b64 = "" 

    return jsonify({
        "prediction": label,
        "confidence": round(confidence, 4),
        "gradcam_image": gradcam_b64,
    })


@app.route("/chat", methods=["POST"])
def chat():
    """Gemini-powered medical assistant chat endpoint.
    Expects JSON body:
    {
        "message": "user's message text",
        "history": [{"role": "user"|"assistant", "text": "..."}],
        "prediction_context": {
            "prediction": "NORMAL"|"PNEUMONIA",
            "confidence": 0.95
        }
    }
    Returns JSON:
    {
        "reply": "Gemini's response text"
    }
    """
    if _gemini_client is None:
        return jsonify({
            "error": "Chat is unavailable. Gemini API key not configured."
        }), 503

    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' in request body."}), 400

    user_message = data["message"].strip()
    if not user_message:
        return jsonify({"error": "Message cannot be empty."}), 400

    prediction_context = data.get("prediction_context", {})
    context_str = ""
    if prediction_context:
        pred = prediction_context.get("prediction", "Unknown")
        conf = prediction_context.get("confidence", 0)
        context_str = (
            f"\n\n[PREDICTION CONTEXT] The AI model classified this chest X-ray as "
            f"**{pred}** with **{conf * 100:.1f}%** confidence. "
            f"The model used is a DenseNet121 deep learning architecture trained on "
            f"chest X-ray images with transfer learning."
        )

    # Build full prompt including history
    history = data.get("history", [])
    full_prompt = SYSTEM_PROMPT + context_str + "\n\n"
    
    for msg in history:
        role = "User" if msg.get("role") == "user" else "Assistant"
        text = msg.get("text", "")
        full_prompt += f"{role}: {text}\n\n"
        
    full_prompt += f"User: {user_message}\n\nAssistant:"
    try:
        response = _gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=full_prompt,
        )
        reply_text = response.text

        return jsonify({"reply": reply_text})

    except Exception as exc:
        print(f"[ERROR] Gemini chat failed: {exc}")
        return jsonify({
            "error": f"Chat failed: {str(exc)}"
        }), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        "error": f"File too large. Maximum allowed size is "
                 f"{MAX_CONTENT_LENGTH // (1024 * 1024)} MB."
    }), 413

if __name__ == "__main__":
    _load_model()
    print("[app] Starting Flask server on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)