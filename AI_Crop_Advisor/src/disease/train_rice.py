"""
src/disease/train_rice.py
Trains and deeply fine-tunes MobileNetV2 on Rice_Leaf_AUG dataset (6 classes:
Bacterial Leaf Blight, Brown Spot, Healthy Rice Leaf, Leaf Blast, Leaf scald, Sheath Blight).
Uses full balanced stratified dataset (up to 400 train, 100 val, 100 test per class)
with data augmentation and adaptive learning rate.
"""

import os
import sys
import json
import shutil
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def build_rice_model(n_classes: int = 6):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*config.IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False

    model = tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(n_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model, base


def main():
    print("=" * 65)
    print("DEEP TRAINING RICE LEAF DISEASE IMAGE MODEL (MOBILENETV2)")
    print("=" * 65)

    root = config.RICE_DISEASE_DIR
    if not os.path.exists(root):
        raise FileNotFoundError(f"Rice dataset directory not found at: {root}")

    classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    n_classes = len(classes)
    print(f"Discovered Rice classes ({n_classes}): {classes}")

    # Save class names
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    with open(config.RICE_DISEASE_CLASSES_PATH, "w") as f:
        json.dump(classes, f, indent=2)

    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []

    np.random.seed(42)
    # Scaled up balanced split: up to 400 train, 100 val, 100 test per class (~3,600 images)
    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(root, cls)
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        np.random.shuffle(files)

        n_train = min(400, int(len(files) * 0.70))
        n_val   = min(100, int(len(files) * 0.15))
        n_test  = min(100, int(len(files) * 0.15))

        for f in files[:n_train]:
            train_paths.append(os.path.join(cls_dir, f))
            train_labels.append(idx)
        for f in files[n_train:n_train + n_val]:
            val_paths.append(os.path.join(cls_dir, f))
            val_labels.append(idx)
        for f in files[n_train + n_val:n_train + n_val + n_test]:
            test_paths.append(os.path.join(cls_dir, f))
            test_labels.append(idx)

    print(f"Rice dataset split: {len(train_paths)} train, {len(val_paths)} val, {len(test_paths)} test")

    def load_and_prep(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, config.IMAGE_SIZE)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        return img, label

    def augment(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, 0.15)
        img = tf.image.random_contrast(img, 0.85, 1.15)
        return img, label

    train_ds = (
        tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
        .shuffle(len(train_paths))
        .map(load_and_prep, num_parallel_calls=tf.data.AUTOTUNE)
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(32)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
        .map(load_and_prep, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(32)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_ds = (
        tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
        .map(load_and_prep, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(32)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Warm-start if model exists, or build fresh
    if os.path.exists(config.RICE_DISEASE_MODEL_PATH):
        print(f"\nLoading existing Rice model from {config.RICE_DISEASE_MODEL_PATH} for deep fine-tuning...")
        model = tf.keras.models.load_model(config.RICE_DISEASE_MODEL_PATH)
        base = model.layers[0]
        history1 = None
    else:
        model, base = build_rice_model(n_classes)
        print("\nPhase 1: Pre-training classification head (base frozen)...")
        history1 = model.fit(train_ds, validation_data=val_ds, epochs=4, verbose=1)

    print("\nPhase 2: Deep fine-tuning top 40 layers of MobileNetV2 on expanded dataset...")
    base.trainable = True
    for layer in base.layers[:-40]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=6e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=6,
        callbacks=callbacks,
        verbose=1
    )

    # Save deeply trained model
    model.save(config.RICE_DISEASE_MODEL_PATH)
    print(f"\nDeep Rice disease model saved to: {config.RICE_DISEASE_MODEL_PATH}")

    # Evaluate on holdout test set
    print("\nEvaluating Rice model on test set...")
    probs = model.predict(test_ds, verbose=1)
    preds = np.argmax(probs, axis=1)
    y_test = np.array(test_labels)

    test_acc = float(np.mean(preds == y_test))
    test_loss = float(-np.mean(np.log(np.clip(probs[np.arange(len(y_test)), y_test], 1e-7, 1.0))))

    report = classification_report(y_test, preds, target_names=classes, output_dict=True)
    print("\n==========================================")
    print(f"Rice Model Test Accuracy: {test_acc*100:.2f}%")
    print(f"Rice Model Test Loss: {test_loss:.4f}")
    print("==========================================")

    results = {
        "test_accuracy": round(test_acc, 4),
        "test_loss": round(test_loss, 4),
        "n_classes": n_classes,
        "classes": classes,
        "classification_report": report
    }

    res_path = os.path.join(config.RESULTS_DIR, "rice_disease_results.json")
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGn", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title(f"Rice Leaf Disease — Deep Model Confusion Matrix (Acc: {test_acc*100:.1f}%)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    cm_path = os.path.join(config.FIGURES_DIR, "rice_disease_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # Plot Training curves
    all_acc = (history1.history["accuracy"] if history1 else []) + history2.history["accuracy"]
    all_val_acc = (history1.history["val_accuracy"] if history1 else []) + history2.history["val_accuracy"]
    all_loss = (history1.history["loss"] if history1 else []) + history2.history["loss"]
    all_val_loss = (history1.history["val_loss"] if history1 else []) + history2.history["val_loss"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(all_acc, label="Train Accuracy", marker="o")
    ax1.plot(all_val_acc, label="Val Accuracy", marker="o")
    ax1.set_title("Rice Model Accuracy (Deep Training)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(all_loss, label="Train Loss", marker="o")
    ax2.plot(all_val_loss, label="Val Loss", marker="o")
    ax2.set_title("Rice Model Loss (Deep Training)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    curve_path = os.path.join(config.FIGURES_DIR, "rice_disease_training_accuracy.png")
    plt.savefig(curve_path, dpi=150)
    plt.close()

    # Copy to static figures
    for f in ["rice_disease_confusion_matrix.png", "rice_disease_training_accuracy.png"]:
        shutil.copy(os.path.join(config.FIGURES_DIR, f), os.path.join(config.STATIC_FIGURES_DIR, f))
    print("Rice plots updated in static/figures!")


if __name__ == "__main__":
    main()
