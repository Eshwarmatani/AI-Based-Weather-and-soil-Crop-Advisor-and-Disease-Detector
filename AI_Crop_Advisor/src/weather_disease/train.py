"""
src/weather_disease/train.py

Phase 5 — Weather-Based Disease Risk Model Training.

Trains 5 classifiers, compares performance on the generated weather-disease
dataset. Saves best model with preprocessor.
"""

import os
import sys
import json
import time
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from src.weather_disease.preprocessing import generate_dataset, validate, preprocess

from sklearn.linear_model  import LogisticRegression
from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm           import SVC
from sklearn.pipeline      import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def get_models(random_state=config.RANDOM_STATE):
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=1000, random_state=random_state)),
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=random_state, n_jobs=-1
        ),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(kernel="rbf", probability=True, random_state=random_state)),
        ]),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, random_state=random_state
        ),
    }


def train_all(X_train, X_test, y_train, y_test, class_names):
    models    = get_models()
    results   = []
    best_f1   = -1
    best_name = None
    best_model = None
    all_preds  = {}

    for name, model in models.items():
        print(f"\n{'-'*50}")
        print(f"Training: {name}")
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0

        y_pred = model.predict(X_test)
        metrics = {
            "model":           name,
            "accuracy":        round(accuracy_score(y_test, y_pred),  4),
            "precision":       round(precision_score(y_test, y_pred, average="weighted"), 4),
            "recall":          round(recall_score(y_test, y_pred,    average="weighted"), 4),
            "f1":              round(f1_score(y_test, y_pred,         average="weighted"), 4),
            "train_time_sec":  round(elapsed, 2),
        }
        print(f"  Accuracy  : {metrics['accuracy']:.4f}")
        print(f"  F1        : {metrics['f1']:.4f}")

        results.append(metrics)
        all_preds[name] = y_pred

        if metrics["f1"] > best_f1:
            best_f1    = metrics["f1"]
            best_name  = name
            best_model = model

    print(f"\n{'='*50}")
    print(f"BEST MODEL: {best_name}  (F1 = {best_f1:.4f})")
    return results, all_preds, best_name, best_model


def save_results(results, best_name, best_model, class_names,
                 X_test, y_test, all_preds):
    os.makedirs(config.MODELS_DIR,   exist_ok=True)
    os.makedirs(config.RESULTS_DIR,  exist_ok=True)
    os.makedirs(config.FIGURES_DIR,  exist_ok=True)

    joblib.dump(best_model, config.WEATHER_MODEL_PATH)
    print(f"Best model saved: {config.WEATHER_MODEL_PATH}")

    results_path = os.path.join(config.RESULTS_DIR, "weather_disease_results.json")
    with open(results_path, "w") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)

    pd.DataFrame(results).to_csv(
        os.path.join(config.RESULTS_DIR, "weather_disease_comparison.csv"), index=False
    )

    # Model comparison chart
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, metric in zip(axes, ["accuracy", "precision", "recall", "f1"]):
        vals  = [r[metric] for r in results]
        names = [r["model"].replace(" ", "\n") for r in results]
        bars  = ax.bar(names, vals, color="darkorange")
        ax.set_title(metric.capitalize())
        ax.set_ylim(0, 1.05)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.suptitle("Weather-Disease Model Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, "weather_disease_model_comparison.png"), dpi=150)
    plt.close()

    # Confusion matrix
    cm  = confusion_matrix(y_test, all_preds[best_name])
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f"Confusion Matrix - {best_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, "weather_disease_confusion_matrix.png"), dpi=150)
    plt.close()

    # Feature importance (tree-based models only)
    try:
        est = best_model.named_steps.get("clf", best_model)
        if hasattr(est, "feature_importances_"):
            features = config.WEATHER_DISEASE_FEATURES if hasattr(config, "WEATHER_DISEASE_FEATURES") \
                       else ["temperature", "humidity", "rainfall", "wind_speed"]
            imp  = est.feature_importances_
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.barh(features, imp, color="darkorange")
            ax.set_title(f"Feature Importance - {best_name}")
            ax.set_xlabel("Importance")
            plt.tight_layout()
            plt.savefig(os.path.join(config.FIGURES_DIR, "weather_feature_importance.png"), dpi=150)
            plt.close()
    except Exception:
        pass


def main():
    print("=" * 60)
    print("PHASE 5 - WEATHER DISEASE RISK MODEL TRAINING")
    print("=" * 60)

    df = generate_dataset()
    validate(df)

    (X_train, X_test, y_train, y_test,
     X_train_scaled, X_test_scaled,
     scaler, le, class_names) = preprocess(df, save=True)

    results, all_preds, best_name, best_model = train_all(
        X_train, X_test, y_train, y_test, class_names
    )

    save_results(results, best_name, best_model, class_names,
                 X_test, y_test, all_preds)

    print("\nPhase 5 - Weather disease model training COMPLETE.")
    print(f"Best model : {best_name}")
    print(f"Model saved: {config.WEATHER_MODEL_PATH}")


if __name__ == "__main__":
    main()
