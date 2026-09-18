
"""
src/disease/train.py

Phase 4 — Plant Disease Detection Model Training (MobileNetV2).

Uses transfer learning. EarlyStopping and ModelCheckpoint prevent overfitting.
Augmentation applied to training set only.
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from src.disease.preprocessing import run_preprocessing, build_tf_datasets

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def build_model(n_classes: int, learning_rate: float = config.LEARNING_RATE):
    """Build MobileNetV2-based transfer learning model."""
    import tensorflow as tf

    base = tf.keras.applications.MobileNetV2(
        input_shape=(*config.IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    # Freeze base initially
    base.trainable = False

    model = tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(base_dir: str | None = None):
    import tensorflow as tf

    print("=" * 60)
    print("PHASE 4 - DISEASE IMAGE MODEL TRAINING")
    print("=" * 60)

    # ── Preprocessing ────────────────────────────────────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test, classes = \
        run_preprocessing(base_dir, save=True)

    n_classes = len(classes)
    print(f"\nClasses ({n_classes}): {classes}")

    train_ds, val_ds, test_ds = build_tf_datasets(
        X_train, X_val, X_test, y_train, y_val, y_test, n_classes
    )

    # ── Build model ──────────────────────────────────────────────────────────
    model = build_model(n_classes)
    model.summary()

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    callbacks_phase1 = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=config.DISEASE_MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3,
            patience=3, min_lr=1e-7, verbose=1
        ),
    ]

    # ── Phase 1: Train head only ─────────────────────────────────────────────
    print("\n[Phase 4a] Training classification head (base frozen)...")
    history1 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=10, callbacks=callbacks_phase1, verbose=1
    )

    # ── Phase 2: Fine-tune top layers of base ────────────────────────────────
    print("\n[Phase 4b] Fine-tuning top 30 layers of MobileNetV2...")
    base_model = model.layers[0]
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    # Keep BatchNormalization layers frozen during fine-tuning for stability
    for layer in base_model.layers[-30:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE / 10),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks_phase2 = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=config.DISEASE_MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3,
            patience=3, min_lr=1e-7, verbose=1
        ),
    ]

    history2 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=config.EPOCHS, callbacks=callbacks_phase2, verbose=1
    )

    # Merge histories
    history = {
        "accuracy":     history1.history.get("accuracy", [])     + history2.history.get("accuracy", []),
        "val_accuracy": history1.history.get("val_accuracy", []) + history2.history.get("val_accuracy", []),
        "loss":         history1.history.get("loss", [])         + history2.history.get("loss", []),
        "val_loss":     history1.history.get("val_loss", [])     + history2.history.get("val_loss", []),
    }

    # ── Evaluate on test set ─────────────────────────────────────────────────
    print("\n[Phase 4c] Evaluating on test set...")
    best_model = None
    if os.path.exists(config.DISEASE_MODEL_PATH):
        try:
            best_model = tf.keras.models.load_model(config.DISEASE_MODEL_PATH)
        except Exception:
            best_model = None

    if best_model is None:
        model.save(config.DISEASE_MODEL_PATH)
        best_model = model

    test_loss, test_acc = best_model.evaluate(test_ds, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss    : {test_loss:.4f}")

    # Predictions for confusion matrix
    y_pred_probs = best_model.predict(test_ds, verbose=0)
    y_pred       = np.argmax(y_pred_probs, axis=1)

    y_test_eval = y_test[:len(y_pred)]
    from sklearn.metrics import classification_report
    report = classification_report(y_test_eval, y_pred, target_names=classes, output_dict=True)

    # Save metrics
    disease_results = {
        "test_accuracy": round(float(test_acc), 4),
        "test_loss":     round(float(test_loss), 4),
        "n_classes":     n_classes,
        "classes":       classes,
        "classification_report": report,
    }
    with open(os.path.join(config.RESULTS_DIR, "disease_model_results.json"), "w") as f:
        json.dump(disease_results, f, indent=2)

    # ── Plots ────────────────────────────────────────────────────────────────
    _plot_history(history)
    _plot_confusion_matrix(y_test_eval, y_pred, classes)

    print("\nPhase 4 - Disease model training COMPLETE.")
    print(f"Model saved: {config.DISEASE_MODEL_PATH}")
    return best_model, classes


def _plot_history(history: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["accuracy"],     label="Train Accuracy")
    ax1.plot(history["val_accuracy"], label="Val Accuracy")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history["loss"],     label="Train Loss")
    ax2.plot(history["val_loss"], label="Val Loss")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, "disease_training_accuracy.png"), dpi=150)
    plt.close()
    print("Training curves saved.")


def _plot_confusion_matrix(y_true, y_pred, classes):
    from sklearn.metrics import confusion_matrix
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title("Disease Detection - Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, "disease_confusion_matrix.png"), dpi=150)
    plt.close()
    print("Confusion matrix saved.")


if __name__ == "__main__":
    train()
