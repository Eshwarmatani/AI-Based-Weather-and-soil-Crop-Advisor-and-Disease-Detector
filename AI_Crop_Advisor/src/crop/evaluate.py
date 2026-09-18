"""
src/crop/evaluate.py — Evaluate the saved Crop Recommendation model on test data.
"""

import os
import sys
import json
import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config


def evaluate_saved_model():
    csv_path = os.path.join(config.BASE_DIR, "data", "crop_recommendation", "Crop_recommendation.csv")
    if not os.path.exists(csv_path):
        csv_path = config.CROP_CSV_PATH
    
    df = pd.read_csv(csv_path)
    X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
    y = df["label"]

    # Recreate the exact hold-out test set
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

    model = joblib.load(config.CROP_MODEL_PATH)
    y_pred = model.predict(X_test)

    # Decode integer predictions to class strings if necessary
    if os.path.exists(config.CROP_CLASSES_PATH):
        import json
        with open(config.CROP_CLASSES_PATH, "r") as f:
            classes = json.load(f)
        if len(y_pred) > 0 and isinstance(y_pred[0], (int, np.integer)):
            y_pred = [classes[i] for i in y_pred]

    acc = accuracy_score(y_test, y_pred)
    f1_m = f1_score(y_test, y_pred, average="macro")
    f1_w = f1_score(y_test, y_pred, average="weighted")

    print(f"Loaded model from: {config.CROP_MODEL_PATH}")
    print(f"Test Accuracy:    {acc*100:.2f}%")
    print(f"Test Macro F1:    {f1_m*100:.2f}%")
    print(f"Test Weighted F1: {f1_w*100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return {"accuracy": acc, "macro_f1": f1_m, "weighted_f1": f1_w}


if __name__ == "__main__":
    evaluate_saved_model()
