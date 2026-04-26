# RespiraCheck — AI-Assisted Chest X-Ray Pneumonia Detection

A production-grade deep learning pipeline for binary classification of chest
X-ray images (NORMAL vs PNEUMONIA) using **two competing backbone architectures**
(DenseNet121 and EfficientNetB3) with two-phase transfer learning, Grad-CAM++
explainability, automatic model selection, and a premium full-stack web interface
featuring an integrated **Gemini 2.5 Flash Medical AI Chatbot**.

---

## Project Structure

```
PROJECT 2/
├── chest_xray/                      # Kaggle dataset (user-provided)
├── data_setup.py                    # Component 1 — Dataset merging & re-splitting
├── model.py                         # Component 2 — DenseNet121 + EfficientNetB3 builders
├── train.py                         # Component 3 — Dual-model training pipeline
├── evaluate.py                      # Component 4 — Dual-model evaluation & comparison
├── gradcam.py                       # Component 5 — Grad-CAM++ explainability
├── app.py                           # Component 6 — Flask REST API (Model inference & Gemini chat)
├── frontend/                        # Component 7 — React frontend (Vite)
├── requirements.txt                 # Python dependencies
├── .env                             # Environment variables (Gemini API Key)
└── README.md                        # This file
```

---

## Architecture Overview

### 1. Deep Learning Pipeline
Both backbones share the same classification head and training strategy:

```
Backbone (ImageNet pre-trained)
    └── GlobalAveragePooling2D
        └── Dense(512, relu)
            └── Dropout(0.5)
                └── Dense(1, sigmoid)   →   NORMAL / PNEUMONIA
```

| | DenseNet121 | EfficientNetB3 |
|---|---|---|
| **Phase 1** | Backbone fully frozen, head trained | Backbone fully frozen, head trained |
| **Phase 2** | Last 30 layers unfrozen (BatchNorm frozen) | Last 30 layers unfrozen (BatchNorm frozen) |
| **Checkpoints** | `best_densenet_phase1/2.keras` | `best_efficientnet_phase1/2.keras` |

The winner is decided by **ROC-AUC** on the test set and written to
`best_model_path.txt` — `app.py` reads this file automatically at startup.

### 2. Explainability & Chatbot
- **Grad-CAM++**: When a prediction is made, `tf-keras-vis` generates a Grad-CAM heatmap highlighting the regions the model focused on.
- **Gemini Assistant**: After prediction, a chat interface appears. The Flask backend sends the model's prediction and confidence as context to `google-genai`, allowing the chatbot to explain the results to the user seamlessly.

---

## Setup Instructions

### 1. Python Environment

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
# Windows
venv\Scripts\activate
# Install dependencies
pip install -r requirements.txt
```

> **Note:** Requires Python 3.10+ and TensorFlow 2.20.0.
> A CUDA-compatible GPU is strongly recommended.

### 2. Gemini API Configuration

To enable the AI Chatbot, create a file named `.env` in the root directory:

```bash
# .env
GEMINI_API_KEY=your_actual_api_key_here
```
*(Get a free API key from Google AI Studio)*

### 3. Dataset Placement

Download the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
dataset from Kaggle and extract it to `chest_xray/`.

---

## Running the Pipeline

Execute the components **strictly in this order**:

### Step 1 — Prepare Data
```bash
python data_setup.py
```
- Merges `train/` + `val/` and performs a stratified 80/20 re-split (`random_state=42`)
- Computes balanced class weights

### Step 2 — Train Both Models
```bash
python train.py
```
- Runs **four training phases** sequentially (frozen backbone vs fine-tuned)
- Saves checkpoints and produces a 4×2 grid comparing training curves.

### Step 3 — Evaluate & Compare
```bash
python evaluate.py
```
- Evaluates on `chest_xray/test/`
- Computes metrics: Accuracy, Recall, Precision, F1, AUC-ROC
- Writes the winning model's path to `best_model_path.txt`

### Step 4 — Grad-CAM (optional standalone test)
```bash
python gradcam.py --image chest_xray/test/PNEUMONIA/person1_virus_6.jpeg --model best_densenet_phase2.keras --output gradcam_output.png
```

---

## Starting the Application

### 1. Start the Flask Backend
```bash
python app.py
```
The Flask server starts on `http://0.0.0.0:5000` and automatically loads the best performing model. It exposes `/predict`, `/chat`, and `/health`.

### 2. Start the React Frontend
```bash
cd frontend
npm install   # If running for the first time
npm run dev
```
Opens the React app on `http://localhost:3000`.

---

## API Reference

### `POST /predict`
Uploads an image, returns prediction, confidence, and Base64-encoded Grad-CAM overlay.
```bash
curl -X POST http://localhost:5000/predict -F "file=@image.jpeg"
```

### `POST /chat`
Accepts conversation history and prediction context to power the Gemini Assistant.
```json
{
  "message": "What does this mean?",
  "history": [{"role": "assistant", "text": "I predict PNEUMONIA..."}],
  "prediction_context": {"prediction": "PNEUMONIA", "confidence": 0.95}
}
```

---

## Key Features (Capstone Release)
- **Automatic Model Selection**: `evaluate.py` picks the best weights.
- **Explainability**: Heatmaps generated via Grad-CAM++ algorithms.
- **Premium Interface**: React + Vite UI with dark-mode glassmorphism styling.
- **AI Integration**: Context-aware `google-genai` chatbot to guide patients.
