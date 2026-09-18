"""
src/disease/finetune_plantvillage.py
Fine-tunes the 15-class MobileNetV2 PlantVillage model with data augmentation,
evaluates on hold-out test images, and updates model results & figures.
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


def main():
    print("=" * 65)
    print("FINE-TUNING PLANTVILLAGE (15-CLASS) MOBILENETV2 MODEL")
    print("=" * 65)

    classes_path = config.DISEASE_CLASSES_PATH
    with open(classes_path) as f:
        classes = json.load(f)
    n_classes = len(classes)
    print(f"Target classes ({n_classes}): {classes}")

    # Load existing model
    print(f"\nLoading base model from {config.DISEASE_MODEL_PATH}...")
    model = tf.keras.models.load_model(config.DISEASE_MODEL_PATH)

    root = config.PLANT_DISEASE_DIR
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []

    np.random.seed(42)
    for idx, cls in enumerate(classes):
        cls_folder = os.path.join(root, cls)
        if not os.path.exists(cls_folder):
            continue
        files = [f for f in os.listdir(cls_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        np.random.shuffle(files)

        # 80 for train, 20 for val, 25 for test per class
        train_files = files[:80]
        val_files   = files[80:100]
        test_files  = files[100:125]

        for f in train_files:
            train_paths.append(os.path.join(cls_folder, f))
            train_labels.append(idx)
        for f in val_files:
            val_paths.append(os.path.join(cls_folder, f))
            val_labels.append(idx)
        for f in test_files:
            test_paths.append(os.path.join(cls_folder, f))
            test_labels.append(idx)

    print(f"Dataset split: {len(train_paths)} train, {len(val_paths)} val, {len(test_paths)} test")

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

    # Unfreeze top layers of base model for subtle fine-tuning
    base_model = model.layers[0] if isinstance(model.layers[0], tf.keras.Model) else None
    if base_model:
        base_model.trainable = True
        for layer in base_model.layers[:-35]:
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\nTraining 4 fine-tuning epochs with learning rate 2e-5...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=4,
        verbose=1
    )

    # Save fine-tuned model
    model.save(config.DISEASE_MODEL_PATH)
    print(f"\nSaved updated model to {config.DISEASE_MODEL_PATH}")

    # Evaluate on test set
    print("\nEvaluating on hold-out test set...")
    test_imgs = []
    for p in test_paths:
        img = tf.io.read_file(p)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, config.IMAGE_SIZE)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        test_imgs.append(img.numpy())

    test_batch = np.array(test_imgs)
    probs = model.predict(test_batch, batch_size=32, verbose=0)
    preds = np.argmax(probs, axis=1)
    y_test = np.array(test_labels)

    test_acc = float(np.mean(preds == y_test))
    test_loss = float(-np.mean(np.log(np.clip(probs[np.arange(len(y_test)), y_test], 1e-7, 1.0))))

    report = classification_report(y_test, preds, target_names=classes, output_dict=True)
    print(f"Final Fine-Tuned Test Accuracy: {test_acc*100:.2f}%")
    print(f"Final Test Loss: {test_loss:.4f}")

    results = {
        "test_accuracy": round(test_acc, 4),
        "test_loss": round(test_loss, 4),
        "n_classes": n_classes,
        "classes": classes,
        "classification_report": report
    }

    res_path = os.path.join(config.RESULTS_DIR, "disease_model_results.json")
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)

    # Plot updated confusion matrix
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title(f"PlantVillage Disease Detection — Confusion Matrix (Acc: {test_acc*100:.1f}%)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    cm_path = os.path.join(config.FIGURES_DIR, "disease_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history.history["accuracy"], label="Train Accuracy")
    ax1.plot(history.history["val_accuracy"], label="Val Accuracy")
    ax1.set_title("Fine-Tuning Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history["loss"], label="Train Loss")
    ax2.plot(history.history["val_loss"], label="Val Loss")
    ax2.set_title("Fine-Tuning Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    curve_path = os.path.join(config.FIGURES_DIR, "disease_training_accuracy.png")
    plt.savefig(curve_path, dpi=150)
    plt.close()

    # Copy to static figures
    for f in ["disease_confusion_matrix.png", "disease_training_accuracy.png"]:
        shutil.copy(os.path.join(config.FIGURES_DIR, f), os.path.join(config.STATIC_FIGURES_DIR, f))
    print("Plots updated and copied to static/figures!")


if __name__ == "__main__":
    main()
