"""
model.py — DenseNet121 & EfficientNetB3 Binary Classifiers for PneumoScan

Provides four builder functions implementing a two-phase transfer-learning
strategy for two backbone architectures:

DenseNet121
    Phase 1  – Fully frozen backbone; head trained from scratch.
    Phase 2  – Last 30 layers unfrozen; low-LR fine-tuning.

EfficientNetB3
    Phase 1  – Fully frozen backbone; identical head trained from scratch.
    Phase 2  – Last 30 layers unfrozen; low-LR fine-tuning.

Both architectures share the same classification head:
    GlobalAveragePooling2D → Dense(512, relu) → Dropout(0.5) → Dense(1, sigmoid)
"""

import tensorflow as tf
from tensorflow.keras.applications import DenseNet121, EfficientNetB3
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# ---------------------------------------------------------------------------
# Phase 1 — Frozen feature-extraction model
# ---------------------------------------------------------------------------

def build_phase1_model() -> Model:
    """Build a DenseNet121-based binary classifier with frozen backbone.

    Architecture
    ------------
    DenseNet121 (frozen) → GAP → Dense(512, relu) → Dropout(0.5) →
    Dense(1, sigmoid)

    Returns
    -------
    tf.keras.Model
        Compiled model ready for Phase 1 training.
    """
    print("[model] Building Phase 1 model (frozen DenseNet121) …")

    try:
        base_model = DenseNet121(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load DenseNet121 pre-trained weights: {exc}"
        ) from exc

    # Freeze all base layers
    for layer in base_model.layers:
        layer.trainable = False

    # Classification head
    x = base_model.output
    x = GlobalAveragePooling2D(name="gap")(x)
    x = Dense(512, activation="relu", name="fc_512")(x)
    x = Dropout(0.5, name="dropout_05")(x)
    output = Dense(1, activation="sigmoid", name="output_sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output, name="PneumoScan_Phase1")

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    trainable = sum(1 for l in model.layers if l.trainable)
    frozen = sum(1 for l in model.layers if not l.trainable)
    print(f"  • Total layers   : {len(model.layers)}")
    print(f"  • Trainable      : {trainable}")
    print(f"  • Frozen          : {frozen}")
    print("[model] Phase 1 model built and compiled.\n")

    return model


# ---------------------------------------------------------------------------
# Phase 2 — Partial fine-tuning
# ---------------------------------------------------------------------------

def build_phase2_model(model: Model) -> Model:
    """Unfreeze the last 30 layers of DenseNet121 for fine-tuning.

    Parameters
    ----------
    model : tf.keras.Model
        A trained Phase 1 model (output of ``build_phase1_model``).

    Returns
    -------
    tf.keras.Model
        Recompiled model ready for Phase 2 fine-tuning.
    """
    print("[model] Building Phase 2 model (unfreezing last 30 layers) …")

    # Identify the DenseNet121 base (first sub-model or all layers before head)
    # DenseNet121 layers are embedded directly in the model.
    # We unfreeze the last 30 layers of the ENTIRE model, which mainly
    # affects the tail of the DenseNet backbone + head (head is already
    # trainable).
    total_layers = len(model.layers)

    # Freeze everything first …
    for layer in model.layers:
        layer.trainable = False

    # … then unfreeze the last 30 layers
    num_unfreeze = min(30, total_layers)
    for layer in model.layers[-num_unfreeze:]:
        # Skip BatchNormalization layers to keep running stats stable
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    trainable = sum(1 for l in model.layers if l.trainable)
    frozen = sum(1 for l in model.layers if not l.trainable)
    print(f"  • Total layers   : {total_layers}")
    print(f"  • Trainable      : {trainable}")
    print(f"  • Frozen          : {frozen}")
    print("[model] Phase 2 model recompiled.\n")

    return model


# ---------------------------------------------------------------------------
# EfficientNetB3 Phase 1 — Frozen feature-extraction model
# ---------------------------------------------------------------------------

def build_efficientnet_phase1_model() -> Model:
    """Build an EfficientNetB3-based binary classifier with frozen backbone.

    Architecture
    ------------
    EfficientNetB3 (frozen) → GAP → Dense(512, relu) → Dropout(0.5) →
    Dense(1, sigmoid)

    Returns
    -------
    tf.keras.Model
        Compiled model ready for Phase 1 training.
    """
    print("[model] Building Phase 1 model (frozen EfficientNetB3) …")

    try:
        base_model = EfficientNetB3(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load EfficientNetB3 pre-trained weights: {exc}"
        ) from exc

    # Freeze all base layers
    for layer in base_model.layers:
        layer.trainable = False

    # Classification head (identical to DenseNet121 variant)
    x = base_model.output
    x = GlobalAveragePooling2D(name="eff_gap")(x)
    x = Dense(512, activation="relu", name="eff_fc_512")(x)
    x = Dropout(0.5, name="eff_dropout_05")(x)
    output = Dense(1, activation="sigmoid", name="eff_output_sigmoid")(x)

    model = Model(
        inputs=base_model.input,
        outputs=output,
        name="PneumoScan_EfficientNet_Phase1",
    )

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    trainable = sum(1 for l in model.layers if l.trainable)
    frozen = sum(1 for l in model.layers if not l.trainable)
    print(f"  • Total layers   : {len(model.layers)}")
    print(f"  • Trainable      : {trainable}")
    print(f"  • Frozen          : {frozen}")
    print("[model] EfficientNetB3 Phase 1 model built and compiled.\n")

    return model


# ---------------------------------------------------------------------------
# EfficientNetB3 Phase 2 — Partial fine-tuning
# ---------------------------------------------------------------------------

def build_efficientnet_phase2_model(model: Model) -> Model:
    """Unfreeze the last 30 layers of EfficientNetB3 for fine-tuning.

    Parameters
    ----------
    model : tf.keras.Model
        A trained EfficientNetB3 Phase 1 model.

    Returns
    -------
    tf.keras.Model
        Recompiled model ready for Phase 2 fine-tuning.
    """
    print("[model] Building EfficientNetB3 Phase 2 model (unfreezing last 30 layers) …")

    total_layers = len(model.layers)

    # Freeze everything first …
    for layer in model.layers:
        layer.trainable = False

    # … then unfreeze the last 30 layers (skip BatchNorm to keep stats stable)
    num_unfreeze = min(30, total_layers)
    for layer in model.layers[-num_unfreeze:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    trainable = sum(1 for l in model.layers if l.trainable)
    frozen = sum(1 for l in model.layers if not l.trainable)
    print(f"  • Total layers   : {total_layers}")
    print(f"  • Trainable      : {trainable}")
    print(f"  • Frozen          : {frozen}")
    print("[model] EfficientNetB3 Phase 2 model recompiled.\n")

    return model


# ---------------------------------------------------------------------------
# Quick smoke-test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Smoke-test: building DenseNet121 Phase 1 model …")
    m = build_phase1_model()
    m.summary(print_fn=lambda s: None)  # suppress for brevity
    print("Smoke-test: converting DenseNet121 to Phase 2 …")
    m = build_phase2_model(m)
    print("DenseNet121 smoke-test passed ✓\n")

    print("Smoke-test: building EfficientNetB3 Phase 1 model …")
    e = build_efficientnet_phase1_model()
    e.summary(print_fn=lambda s: None)
    print("Smoke-test: converting EfficientNetB3 to Phase 2 …")
    e = build_efficientnet_phase2_model(e)
    print("EfficientNetB3 smoke-test passed ✓")
