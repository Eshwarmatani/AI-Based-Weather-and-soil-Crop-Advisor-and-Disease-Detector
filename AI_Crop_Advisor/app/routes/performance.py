"""app/routes/performance.py — Model performance page and API."""
import os
import json
from flask import Blueprint, render_template, jsonify
import config

performance_bp = Blueprint("performance", __name__)


def _load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _standardize_performance_data(data, default_model_name, domain_title):
    if not data:
        return None

    classes = data.get("classes", [])
    report = data.get("classification_report", {})
    per_class = []

    if report and classes:
        for c in classes:
            if c in report:
                metrics = report[c]
                per_class.append({
                    "class": c,
                    "precision": round(float(metrics.get("precision", 0)), 4),
                    "recall": round(float(metrics.get("recall", 0)), 4),
                    "f1_score": round(float(metrics.get("f1-score", metrics.get("f1", 0))), 4),
                    "support": int(metrics.get("support", 0))
                })

    macro_avg = report.get("macro avg", {}) if isinstance(report, dict) else {}
    weighted_avg = report.get("weighted avg", {}) if isinstance(report, dict) else {}

    test_samples = data.get("total_test_samples")
    if test_samples is None and macro_avg.get("support") is not None:
        test_samples = int(macro_avg["support"])
    elif test_samples is None and per_class:
        test_samples = sum(p["support"] for p in per_class)

    accuracy = data.get("test_accuracy")
    if accuracy is None and "accuracy" in report:
        accuracy = round(float(report["accuracy"]), 4)

    bal_acc = data.get("balanced_accuracy")
    if bal_acc is not None:
        bal_acc = round(float(bal_acc), 4)

    macro_p = data.get("test_precision_macro") or macro_avg.get("precision")
    macro_r = data.get("test_recall_macro") or macro_avg.get("recall")
    macro_f1 = data.get("test_f1_macro") or macro_avg.get("f1-score")

    weighted_p = data.get("test_precision_weighted") or weighted_avg.get("precision")
    weighted_r = data.get("test_recall_weighted") or weighted_avg.get("recall")
    weighted_f1 = data.get("test_f1_weighted") or weighted_avg.get("f1-score")

    standardized = dict(data)
    standardized.update({
        "domain_title": domain_title,
        "best_model": data.get("best_model", default_model_name),
        "classes": classes,
        "n_classes": len(classes) if classes else len(per_class),
        "total_test_samples": test_samples,
        "test_accuracy": round(float(accuracy), 4) if accuracy is not None else None,
        "balanced_accuracy": bal_acc,
        "test_precision_macro": round(float(macro_p), 4) if macro_p is not None else None,
        "test_recall_macro": round(float(macro_r), 4) if macro_r is not None else None,
        "test_f1_macro": round(float(macro_f1), 4) if macro_f1 is not None else None,
        "test_precision_weighted": round(float(weighted_p), 4) if weighted_p is not None else None,
        "test_recall_weighted": round(float(weighted_r), 4) if weighted_r is not None else None,
        "test_f1_weighted": round(float(weighted_f1), 4) if weighted_f1 is not None else None,
        "per_class": per_class,
        "confusion_matrix": data.get("confusion_matrix", []),
        "matrix_labels": data.get("matrix_labels", classes),
    })
    return standardized


@performance_bp.route("/model-performance")
def model_performance():
    return render_template("model_performance.html")


@performance_bp.route("/api/performance/crop")
def api_crop_performance():
    data = _load_json(os.path.join(config.RESULTS_DIR, "crop_model_results.json"))
    if not data:
        return jsonify({"error": "Crop model not trained yet. Run src/crop/train.py"}), 404

    # Ensure results array exists and matches comparison contract
    comp_csv = os.path.join(config.RESULTS_DIR, "crop_model_comparison.csv")
    times = {}
    if os.path.exists(comp_csv):
        try:
            import pandas as pd
            df_csv = pd.read_csv(comp_csv)
            for _, row in df_csv.iterrows():
                times[row["model"]] = row.get("train_time_sec", "—")
        except Exception:
            pass

    if "results" not in data and "validation_comparison" in data:
        data["results"] = [
            {
                "model": item["model"],
                "accuracy": item.get("val_accuracy", item.get("accuracy", 0)),
                "precision": item.get("val_precision_macro", item.get("precision", 0)),
                "recall": item.get("val_recall_macro", item.get("recall", 0)),
                "f1": item.get("val_f1_macro", item.get("f1", 0)),
                "train_time_sec": times.get(item["model"], item.get("train_time_sec", "—")),
            }
            for item in data["validation_comparison"]
        ]
    elif "results" in data and times:
        for r in data["results"]:
            if "train_time_sec" not in r or r["train_time_sec"] == "—":
                r["train_time_sec"] = times.get(r.get("model"), "—")

    standardized = _standardize_performance_data(data, "Random Forest", "Crop Recommendation")
    return jsonify(standardized)


@performance_bp.route("/api/performance/disease")
def api_disease_performance():
    data = _load_json(os.path.join(config.RESULTS_DIR, "disease_model_results.json"))
    if not data:
        return jsonify({"error": "Disease model not trained yet. Run src/disease/train.py"}), 404
    standardized = _standardize_performance_data(data, "MobileNetV2 (Fine-Tuned)", "Plant Disease Detection (Garden Crops)")
    return jsonify(standardized)


@performance_bp.route("/api/performance/weather")
def api_weather_performance():
    data = _load_json(os.path.join(config.RESULTS_DIR, "weather_disease_results.json"))
    if not data:
        return jsonify({"error": "Weather model not trained yet. Run src/weather_disease/train.py"}), 404
    standardized = _standardize_performance_data(data, "Support Vector Machine", "Weather-Based Disease Risk")
    return jsonify(standardized)


@performance_bp.route("/api/performance/rice-disease")
def api_rice_disease_performance():
    data = _load_json(os.path.join(config.RESULTS_DIR, "rice_disease_results.json"))
    if not data:
        return jsonify({"error": "Rice disease model not trained yet. Run src/disease/train_rice.py"}), 404
    standardized = _standardize_performance_data(data, "MobileNetV2 (Deep Fine-Tuned)", "Rice Leaf Disease Detection")
    return jsonify(standardized)
