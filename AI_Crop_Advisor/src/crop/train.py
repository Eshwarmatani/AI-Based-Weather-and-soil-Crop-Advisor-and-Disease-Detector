"""
src/crop/train.py - Train, tune, compare, and evaluate Crop Recommendation models.
Adheres strictly to zero-leakage, real-data, and scientific evaluation guidelines.
"""

import os
import sys
import json
import time
import shutil
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

from typing import Any, cast

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config

RANDOM_STATE = 42
ZERO_DIVISION: Any = 0


def load_and_inspect_data():
    csv_path = os.path.join(config.BASE_DIR, "data", "crop_recommendation", "Crop_recommendation.csv")
    if not os.path.exists(csv_path):
        csv_path = config.CROP_CSV_PATH
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Crop dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded Crop Dataset from: {csv_path}")
    print(f"Shape: {df.shape}")
    print(f"Features: {[c for c in df.columns if c != 'label']}")
    print(f"Target classes ({df['label'].nunique()}): {sorted(df['label'].unique())}")
    assert df.isnull().sum().sum() == 0, "Missing values detected in Crop dataset!"
    return df


def train_and_evaluate_all():
    df = load_and_inspect_data()

    feature_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    X = df[feature_cols].copy()
    y_raw = df["label"].copy()

    # Encode target labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw.to_numpy())
    assert le.classes_ is not None
    classes: list[str] = [str(c) for c in le.classes_]

    # 1. Stratified 70% Train, 15% Val, 15% Test Split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
    )
    val_rel_size = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_rel_size, random_state=RANDOM_STATE, stratify=y_train_val
    )

    print(f"Data Split: Train={len(X_train)} (70%), Val={len(X_val)} (15%), Test={len(X_test)} (15%)")

    # Verify no overlap
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_val, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert len(set(X_train.index).intersection(set(X_test.index))) == 0, "Train-Test Leakage detected!"
    assert len(set(X_val.index).intersection(set(X_test.index))) == 0, "Val-Test Leakage detected!"

    # Save fitted scaler for standalone preprocessor artifact
    standalone_scaler = StandardScaler()
    standalone_scaler.fit(cast(Any, X_train))
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    joblib.dump(standalone_scaler, config.CROP_PREPROCESSOR_PATH)
    print(f"Saved preprocessor to: {config.CROP_PREPROCESSOR_PATH}")

    # Model definitions with Pipelines to prevent scaling leakage
    model_configs = {
        "Logistic Regression": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
            ]),
            "params": {
                "clf__C": [0.1, 1.0, 10.0]
            }
        },
        "Decision Tree": {
            "pipeline": Pipeline([
                ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))
            ]),
            "params": {
                "clf__max_depth": [None, 10, 20],
                "clf__min_samples_split": [2, 5]
            }
        },
        "Random Forest": {
            "pipeline": Pipeline([
                ("clf", RandomForestClassifier(random_state=RANDOM_STATE))
            ]),
            "params": {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [None, 15, 25]
            }
        },
        "KNN": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier())
            ]),
            "params": {
                "clf__n_neighbors": [3, 5, 7, 9],
                "clf__weights": ["uniform", "distance"]
            }
        },
        "SVM": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(probability=True, random_state=RANDOM_STATE))
            ]),
            "params": {
                "clf__C": [0.5, 1.0, 5.0],
                "clf__kernel": ["rbf", "linear"]
            }
        },
        "Gradient Boosting": {
            "pipeline": Pipeline([
                ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE))
            ]),
            "params": {
                "clf__n_estimators": [100, 150],
                "clf__learning_rate": [0.05, 0.1],
                "clf__max_depth": [3, 5]
            }
        }
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    val_comparison = []
    trained_estimators = {}

    print("\n--- Training, Cross-Validation & Tuning on Training Data ---")
    for name, cfg in model_configs.items():
        print(f"Tuning {name} with 5-fold Stratified CV...")
        t0 = time.time()
        grid = GridSearchCV(
            cfg["pipeline"],
            cfg["params"],
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
            refit=True
        )
        grid.fit(cast(Any, X_train), y_train)
        elapsed = time.time() - t0
        best_est = grid.best_estimator_
        trained_estimators[name] = best_est

        # Evaluate on Validation Set (completely untouched during CV search)
        y_val_pred = best_est.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        val_p_macro = precision_score(y_val, y_val_pred, average="macro", zero_division=ZERO_DIVISION)
        val_r_macro = recall_score(y_val, y_val_pred, average="macro", zero_division=ZERO_DIVISION)
        val_f1_macro = f1_score(y_val, y_val_pred, average="macro", zero_division=ZERO_DIVISION)
        val_f1_weighted = f1_score(y_val, y_val_pred, average="weighted", zero_division=ZERO_DIVISION)

        val_comparison.append({
            "model": name,
            "best_params": str(grid.best_params_),
            "cv_best_score": round(float(grid.best_score_), 4),
            "val_accuracy": round(float(val_acc), 4),
            "val_precision_macro": round(float(val_p_macro), 4),
            "val_recall_macro": round(float(val_r_macro), 4),
            "val_f1_macro": round(float(val_f1_macro), 4),
            "val_f1_weighted": round(float(val_f1_weighted), 4),
            "train_time_sec": round(elapsed, 2),
        })
        print(f"  {name}: CV Macro F1 = {grid.best_score_:.4f}, Val Accuracy = {val_acc:.4f}, Val Macro F1 = {val_f1_macro:.4f}, Time = {elapsed:.2f}s")

    comp_df = pd.DataFrame(val_comparison)
    print("\nValidation Comparison Table:")
    print(comp_df[["model", "val_accuracy", "val_precision_macro", "val_recall_macro", "val_f1_macro"]].to_string(index=False))

    # Select best model based on validation macro F1 (and tie-break with accuracy)
    best_row = comp_df.sort_values(by=["val_f1_macro", "val_accuracy"], ascending=[False, False]).iloc[0]
    best_model_name = best_row["model"]
    best_estimator = trained_estimators[best_model_name]
    print(f"\nSelected Best Model: {best_model_name} (Val Macro F1 = {best_row['val_f1_macro']:.4f})")

    # 2. Final Evaluation ONCE on untouched Test Set
    print("\n--- Final Evaluation on Untouched Holdout Test Set ---")
    y_test_pred = best_estimator.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_p_macro = precision_score(y_test, y_test_pred, average="macro", zero_division=ZERO_DIVISION)
    test_r_macro = recall_score(y_test, y_test_pred, average="macro", zero_division=ZERO_DIVISION)
    test_f1_macro = f1_score(y_test, y_test_pred, average="macro", zero_division=ZERO_DIVISION)
    test_f1_weighted = f1_score(y_test, y_test_pred, average="weighted", zero_division=ZERO_DIVISION)

    print(f"Test Accuracy:          {test_acc*100:.2f}%")
    print(f"Test Macro Precision:   {test_p_macro*100:.2f}%")
    print(f"Test Macro Recall:      {test_r_macro*100:.2f}%")
    print(f"Test Macro F1:          {test_f1_macro*100:.2f}%")
    print(f"Test Weighted F1:       {test_f1_weighted*100:.2f}%")

    test_report = classification_report(y_test, y_test_pred, target_names=classes, output_dict=True)

    # Save final model
    joblib.dump(best_estimator, config.CROP_MODEL_PATH)
    with open(config.CROP_CLASSES_PATH, "w") as f:
        json.dump(classes, f, indent=2)
    print(f"Saved final model to: {config.CROP_MODEL_PATH}")
    print(f"Saved class names to: {config.CROP_CLASSES_PATH}")

    # Save CSV comparison report
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    os.makedirs(config.STATIC_FIGURES_DIR, exist_ok=True)

    comp_csv_path = os.path.join(config.RESULTS_DIR, "crop_model_results.csv")
    comp_df.to_csv(comp_csv_path, index=False)
    print(f"Saved results CSV to: {comp_csv_path}")

    # Also save standard comparison CSV expected by performance dashboard
    comp_dash_path = os.path.join(config.RESULTS_DIR, "crop_model_comparison.csv")
    dash_df = pd.DataFrame([
        {
            "model": r["model"],
            "accuracy": r["val_accuracy"],
            "precision": r["val_precision_macro"],
            "recall": r["val_recall_macro"],
            "f1": r["val_f1_macro"],
            "train_time_sec": r.get("train_time_sec", 0.0),
        }
        for r in val_comparison
    ])
    dash_df.to_csv(comp_dash_path, index=False)
    print(f"Saved comparison CSV to: {comp_dash_path}")

    # Save comprehensive JSON results
    results_summary = [
        {
            "model": r["model"],
            "accuracy": r["val_accuracy"],
            "precision": r["val_precision_macro"],
            "recall": r["val_recall_macro"],
            "f1": r["val_f1_macro"],
            "train_time_sec": r.get("train_time_sec", 0.0),
        }
        for r in val_comparison
    ]
    final_results = {
        "best_model": best_model_name,
        "test_accuracy": round(float(test_acc), 4),
        "test_precision_macro": round(float(test_p_macro), 4),
        "test_recall_macro": round(float(test_r_macro), 4),
        "test_f1_macro": round(float(test_f1_macro), 4),
        "test_f1_weighted": round(float(test_f1_weighted), 4),
        "n_classes": len(classes),
        "classes": classes,
        "total_test_samples": len(y_test),
        "confusion_matrix": cm.tolist(),
        "matrix_labels": classes,
        "results": results_summary,
        "validation_comparison": val_comparison,
        "classification_report": test_report
    }
    with open(os.path.join(config.RESULTS_DIR, "crop_model_results.json"), "w") as f:
        json.dump(final_results, f, indent=2)

    # 3. Generate Plots
    # A. Model Comparison Bar Chart
    plt.figure(figsize=(10, 5))
    x = np.arange(len(comp_df))
    width = 0.35
    plt.bar(x - width/2, comp_df["val_accuracy"] * 100, width, label="Val Accuracy %", color="#22c55e")
    plt.bar(x + width/2, comp_df["val_f1_macro"] * 100, width, label="Val Macro F1 %", color="#3b82f6")
    plt.xticks(x, [str(m) for m in comp_df["model"]], rotation=20, ha="right")
    plt.ylabel("Percentage (%)")
    plt.ylim(80, 102)
    plt.title("Crop Recommendation - Model Validation Comparison")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    cmp_plot_path = os.path.join(config.FIGURES_DIR, "crop_model_comparison.png")
    plt.savefig(cmp_plot_path, dpi=150)
    plt.close()

    # B. Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGn", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title(f"Crop Recommendation - Test Confusion Matrix ({best_model_name}, Acc: {test_acc*100:.2f}%)")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    cm_plot_path = os.path.join(config.FIGURES_DIR, "crop_confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=150)
    plt.close()

    # Sync to static figures for web UI
    for f in ["crop_model_comparison.png", "crop_confusion_matrix.png"]:
        src_f = os.path.join(config.FIGURES_DIR, f)
        dst_f = os.path.join(config.STATIC_FIGURES_DIR, f)
        if os.path.exists(src_f):
            shutil.copy2(src_f, dst_f)
    print("Figures saved to reports/figures/ and static/figures/")

    return final_results


def main():
    """Main entry point to train and evaluate all crop models."""
    return train_and_evaluate_all()


if __name__ == "__main__":
    main()
