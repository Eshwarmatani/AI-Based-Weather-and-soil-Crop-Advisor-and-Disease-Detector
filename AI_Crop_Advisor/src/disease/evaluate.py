"""
src/disease/evaluate.py — Evaluate saved Plant Disease MobileNetV2 model on unseen test set.
Computes overall and per-class metrics, confusion matrix, error analysis, and exports CSV.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config


def evaluate_saved_model():
    print("=" * 65)
    print("EVALUATION: PLANT DISEASE DETECTION MODEL (MOBILENETV2)")
    print("=" * 65)

    if not os.path.exists(config.DISEASE_MODEL_PATH):
        raise FileNotFoundError(f"Disease model not found at: {config.DISEASE_MODEL_PATH}")

    with open(config.DISEASE_CLASSES_PATH) as f:
        classes = json.load(f)
    n_classes = len(classes)
    print(f"Loaded {n_classes} target classes from: {config.DISEASE_CLASSES_PATH}")

    model = tf.keras.models.load_model(config.DISEASE_MODEL_PATH)
    print(f"Loaded trained MobileNetV2 model from: {config.DISEASE_MODEL_PATH}")

    root = config.PLANT_DISEASE_DIR
    test_paths, test_labels = [], []

    np.random.seed(42)
    for idx, cls in enumerate(classes):
        cls_folder = os.path.join(root, cls)
        if not os.path.exists(cls_folder):
            continue
        files = [f for f in os.listdir(cls_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        np.random.shuffle(files)
        # Exactly matching hold-out test split (25 samples per class)
        for f in files[100:125]:
            test_paths.append(os.path.join(cls_folder, f))
            test_labels.append(idx)

    print(f"Evaluating on {len(test_paths)} hold-out test images...")

    def load_and_prep(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, config.IMAGE_SIZE)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        return img, label

    test_ds = (
        tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
        .map(load_and_prep, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(32)
        .prefetch(tf.data.AUTOTUNE)
    )

    probs = model.predict(test_ds, verbose=1)
    preds = np.argmax(probs, axis=1)
    y_test = np.array(test_labels)

    acc = float(np.mean(preds == y_test))
    report_dict = classification_report(y_test, preds, target_names=classes, output_dict=True)
    report_text = classification_report(y_test, preds, target_names=classes)

    macro_f1 = report_dict["macro avg"]["f1-score"]
    weighted_f1 = report_dict["weighted avg"]["f1-score"]
    macro_p = report_dict["macro avg"]["precision"]
    macro_r = report_dict["macro avg"]["recall"]

    print("\n" + "=" * 50)
    print(f"Test Accuracy:         {acc*100:.2f}%")
    print(f"Test Macro Precision:  {macro_p*100:.2f}%")
    print(f"Test Macro Recall:     {macro_r*100:.2f}%")
    print(f"Test Macro F1:         {macro_f1*100:.2f}%")
    print(f"Test Weighted F1:      {weighted_f1*100:.2f}%")
    print("=" * 50)
    print("\nClassification Report:\n", report_text)

    # ── Error Analysis ──────────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, preds)
    print("\n--- Error Analysis: Confused Classes ---")
    confused = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                confused.append((classes[i], classes[j], cm[i, j]))
                print(f"  Actual: '{classes[i]}'  -->  Predicted as: '{classes[j]}' ({cm[i, j]} samples)")
    if not confused:
        print("  Zero misclassifications on the test set!")

    # ── Save CSV Report ────────────────────────────────────────────────────────
    rows = []
    for cls in classes:
        rows.append({
            "class": cls,
            "precision": round(report_dict[cls]["precision"], 4),
            "recall": round(report_dict[cls]["recall"], 4),
            "f1_score": round(report_dict[cls]["f1-score"], 4),
            "support": int(report_dict[cls]["support"])
        })
    csv_df = pd.DataFrame(rows)
    csv_df.loc[len(csv_df)] = {
        "class": "OVERALL_MACRO",
        "precision": round(macro_p, 4),
        "recall": round(macro_r, 4),
        "f1_score": round(macro_f1, 4),
        "support": len(y_test)
    }
    csv_df.loc[len(csv_df)] = {
        "class": "OVERALL_ACCURACY",
        "precision": round(acc, 4),
        "recall": round(acc, 4),
        "f1_score": round(acc, 4),
        "support": len(y_test)
    }

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(config.RESULTS_DIR, "disease_model_results.csv")
    csv_df.to_csv(csv_path, index=False)
    print(f"\nSaved disease evaluation CSV to: {csv_path}")

    # ── Save Confusion Matrix Figure ───────────────────────────────────────────
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGn", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title(f"Plant Disease Detection — Test Confusion Matrix (MobileNetV2, Acc: {acc*100:.2f}%)")
    ax.set_xlabel("Predicted Condition")
    ax.set_ylabel("True Condition")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = os.path.join(config.FIGURES_DIR, "disease_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # Also save training curves if available or sync
    curve_src = os.path.join(config.FIGURES_DIR, "disease_training_accuracy.png")
    curve_dst = os.path.join(config.FIGURES_DIR, "disease_training_curves.png")
    if os.path.exists(curve_src) and not os.path.exists(curve_dst):
        import shutil
        shutil.copy2(curve_src, curve_dst)

    # Sync to static figures
    for f in ["disease_confusion_matrix.png", "disease_training_curves.png", "disease_training_accuracy.png"]:
        src_f = os.path.join(config.FIGURES_DIR, f)
        dst_f = os.path.join(config.STATIC_FIGURES_DIR, f)
        if os.path.exists(src_f):
            import shutil
            shutil.copy2(src_f, dst_f)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report_dict,
        "confused_classes": confused
    }


if __name__ == "__main__":
    evaluate_saved_model()
