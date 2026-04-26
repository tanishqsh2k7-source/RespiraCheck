# PneumoScan — AI-Assisted Chest X-Ray Pneumonia Detection

A production-grade deep learning pipeline for binary classification of chest
X-ray images (NORMAL vs PNEUMONIA) using **two competing backbone architectures**
(DenseNet121 and EfficientNetB3) with two-phase transfer learning, Grad-CAM++
explainability, automatic model selection, and a full-stack web interface.

---

## Project Structure

```
PROJECT 2/
├── chest_xray/                      # Kaggle dataset (user-provided)
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── val/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test/
│       ├── NORMAL/
│       └── PNEUMONIA/
│
├── data_setup.py                    # Component 1 — Dataset merging & re-splitting
├── model.py                         # Component 2 — DenseNet121 + EfficientNetB3 builders
├── train.py                         # Component 3 — Dual-model training pipeline
├── evaluate.py                      # Component 4 — Dual-model evaluation & comparison
├── gradcam.py                       # Component 5 — Grad-CAM++ explainability
├── app.py                           # Component 6 — Flask REST API
├── App.jsx                          # Component 7 — React frontend component
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

### Generated artefacts (after running each step)

#### After `data_setup.py`
```
├── train_files.pkl                  # (filepath, label) tuples — train split
├── val_files.pkl                    # (filepath, label) tuples — val split
└── class_weights.pkl                # {0: w_normal, 1: w_pneumonia}
```

#### After `train.py`
```
├── best_densenet_phase1.keras       # DenseNet121 best Phase 1 checkpoint (val_auc)
├── best_densenet_phase2.keras       # DenseNet121 best Phase 2 checkpoint (val_auc)
├── best_efficientnet_phase1.keras   # EfficientNetB3 best Phase 1 checkpoint (val_auc)
├── best_efficientnet_phase2.keras   # EfficientNetB3 best Phase 2 checkpoint (val_auc)
├── phase1_history.pkl               # DenseNet121 Phase 1 training history
├── phase2_history.pkl               # DenseNet121 Phase 2 training history
├── efficientnet_phase1_history.pkl  # EfficientNetB3 Phase 1 training history
├── efficientnet_phase2_history.pkl  # EfficientNetB3 Phase 2 training history
└── training_curves_comparison.png   # 4×2 side-by-side comparison plot
```

#### After `evaluate.py`
```
├── confusion_matrix_densenet.png    # Confusion matrix — DenseNet121
├── confusion_matrix_efficientnet.png# Confusion matrix — EfficientNetB3
├── roc_curve_comparison.png         # Both ROC curves on one figure
├── pr_curve_comparison.png          # Both PR curves on one figure
├── evaluation_report.txt            # Full report + side-by-side comparison table
└── best_model_path.txt              # Path to winning model (read by app.py)
```

---

## Architecture Overview

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
| **Phase 2** | Last 30 layers unfrozen (BatchNorm stays frozen) | Last 30 layers unfrozen (BatchNorm stays frozen) |
| **LR** | 1e-4 → 1e-5 | 1e-4 → 1e-5 |
| **Checkpoints** | `best_densenet_phase1/2.keras` | `best_efficientnet_phase1/2.keras` |

The winner is decided by **ROC-AUC** on the test set and written to
`best_model_path.txt` — `app.py` reads this file automatically at startup.

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
> A CUDA-compatible GPU is strongly recommended (TF 2.20 supports CUDA 12.x).
> The `tf_keras` package is installed automatically from `requirements.txt` and
> provides the legacy Keras 2 API that `tf-keras-vis` requires under TF 2.16+.

### 2. Dataset Placement

Download the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
dataset from Kaggle and extract it so the folder structure matches:

```
PROJECT 2/
└── chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

> **Why merge val/?** The default `val/` split contains only 16 images per
> class — far too small to be statistically meaningful.  `data_setup.py` merges
> `train/` + `val/` and re-splits them 80/20 with stratification.

### 3. React Frontend (optional)

```bash
# In a separate terminal, scaffold a React project
npx -y create-react-app frontend
cd frontend
copy ..\App.jsx src\App.js # Windows
npm start
```

---

## Running the Pipeline

Execute the components **strictly in this order**:

### Step 1 — Prepare Data

```bash
python data_setup.py
```

- Merges `train/` + `val/` (removes the tiny 16-image-per-class default split)
- Performs a stratified 80/20 re-split (`random_state=42`)
- Computes balanced class weights for the 1:2.7 imbalance
- Saves `train_files.pkl`, `val_files.pkl`, `class_weights.pkl`

---

### Step 2 — Train Both Models

```bash
python train.py
```

Runs **four training phases** sequentially:

| Phase | Model | Strategy | Epochs |
|---|---|---|---|
| 1 | DenseNet121 | Frozen backbone | 10 |
| 2 | DenseNet121 | Unfreeze last 30 layers | 10 |
| 3 | EfficientNetB3 | Frozen backbone | 10 |
| 4 | EfficientNetB3 | Unfreeze last 30 layers | 10 |

All phases use:
- Augmentation: rotation ±20°, zoom 10%, shifts 10%, horizontal flip
- Callbacks: EarlyStopping (patience 5), ModelCheckpoint (best val_auc), ReduceLROnPlateau
- Class weights loaded from `class_weights.pkl`

Produces `training_curves_comparison.png` — a 4×2 grid comparing both models
across both phases (blue = DenseNet121, orange = EfficientNetB3).

---

### Step 3 — Evaluate & Compare

```bash
python evaluate.py
```

- Loads **both** Phase 2 best checkpoints and evaluates on `chest_xray/test/`
- Prints a side-by-side comparison table:

```
Metric                         |    DenseNet121 |  EfficientNetB3
------------------------------------------------------------
Test Accuracy                  |         0.XXXX |          0.XXXX
Pneumonia Recall               |         0.XXXX |          0.XXXX
Pneumonia Precision            |         0.XXXX |          0.XXXX
F1-Score (Pneumonia)           |         0.XXXX |          0.XXXX
AUC-ROC                        |         0.XXXX |          0.XXXX
Avg Precision (AP)             |         0.XXXX |          0.XXXX
Test Loss                      |         0.XXXX |          0.XXXX
```

- Saves `roc_curve_comparison.png` and `pr_curve_comparison.png` (both models on same axes)
- Writes the winning model's path to `best_model_path.txt`
- Prints final verdict: `Best Model: DenseNet121 / EfficientNetB3 based on AUC-ROC score`

---

### Step 4 — Grad-CAM (optional standalone)

```bash
python gradcam.py --image chest_xray/test/PNEUMONIA/person1_virus_6.jpeg --model best_densenet_phase2.keras --output gradcam_output.png
```
2
Saves a side-by-side `[Original X-ray | Grad-CAM++ Overlay]` figure with the
predicted label and confidence in the title.

---

### Step 5 — Start the API Server

```bash
python app.py
```

The Flask server starts on `http://0.0.0.0:5000`.

**Model auto-selection:** At startup `app.py` reads `best_model_path.txt`
(written by `evaluate.py`).  If the file does not exist it falls back to the
`MODEL_PATH` environment variable, then to `best_densenet_phase2.keras`.

```bash
# Override manually if needed
MODEL_PATH=best_efficientnet_phase2.keras python app.py
```

---

### Step 6 — Start the Frontend

```bash
cd frontend
npm start
```

Opens the React app on `http://localhost:3000`.

---

## API Reference

### `GET /health`

```bash
curl http://localhost:5000/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "best_efficientnet_phase2.keras"
}
```

### `POST /predict`

```bash
curl -X POST http://localhost:5000/predict \
  -F "file=@chest_xray/test/PNEUMONIA/person1_virus_6.jpeg"
```

```json
{
  "prediction": "PNEUMONIA",
  "confidence": 0.9732,
  "gradcam_image": "<base64-encoded PNG>"
}
```

**Validation rules enforced by the API:**
- File key must be `file`
- Extensions: `.jpg`, `.jpeg`, `.png` only
- Maximum file size: **5 MB**

**Error responses:**

| HTTP | Scenario |
|---|---|
| `400` | Missing file / wrong extension / corrupt image |
| `413` | File exceeds 5 MB |
| `503` | Model failed to load at startup |

---

## Key Upgrades over Semester 3 Prototype

| Aspect | Semester 3 (Prototype) | Semester 4 (Capstone) |
|---|---|---|
| Backbone | ResNet50 (frozen) | DenseNet121 **+** EfficientNetB3 (two-phase fine-tuning) |
| Model selection | Manual | Automatic — `best_model_path.txt` written by `evaluate.py` |
| Validation split | 16 images/class (default) | 20% stratified re-split |
| Class imbalance | None | `compute_class_weight` (balanced) |
| Explainability | None | Grad-CAM++ via `tf-keras-vis` |
| Training curves | N/A | 4×2 dual-model comparison plot |
| Evaluation | Accuracy only | Accuracy, Recall, Precision, F1, AUC-ROC, AP, CM, ROC+PR curves |
| Target accuracy | 74.84% | > 80% |
| Pneumonia recall | 91% | > 93% |
| Deployment | Notebook only | Flask API + React frontend |

