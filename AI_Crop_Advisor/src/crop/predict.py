"""
src/crop/predict.py

Crop Recommendation inference module.
Loads the saved model and returns Top-3 predictions with probabilities.
"""

import os
import sys
import json
import numpy as np
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

_model       = None
_class_names = None


def _load():
    """Lazy-load model and class names once."""
    global _model, _class_names

    if _model is None:
        if not os.path.exists(config.CROP_MODEL_PATH):
            raise FileNotFoundError(
                f"Crop model not found at {config.CROP_MODEL_PATH}. "
                "Run src/crop/train.py first."
            )
        _model = joblib.load(config.CROP_MODEL_PATH)

    if _class_names is None:
        classes_path = os.path.join(config.MODELS_DIR, "crop_classes.json")
        with open(classes_path, "r") as f:
            _class_names = json.load(f)


def predict_top3(N: float, P: float, K: float,
                 temperature: float, humidity: float,
                 ph: float, rainfall: float) -> list[dict]:
    """
    Predict the top-3 most suitable crops.

    Parameters
    ----------
    N, P, K         : Soil nutrient values
    temperature     : Temperature in °C
    humidity        : Relative humidity %
    ph              : Soil pH
    rainfall        : Rainfall in mm

    Returns
    -------
    List of dicts (sorted by probability, descending):
        [
          {"rank": 1, "crop": "rice", "probability": 0.94, "confidence": "High"},
          ...
        ]
    """
    _load()

    import pandas as pd
    features = pd.DataFrame(
        [[N, P, K, temperature, humidity, ph, rainfall]],
        columns=config.CROP_FEATURES
    )

    if hasattr(_model, "predict_proba"):
        probs = _model.predict_proba(features)[0]
    else:
        # Fallback: one-hot from predict
        pred_idx  = _model.predict(features)[0]
        probs      = np.zeros(len(_class_names))
        probs[pred_idx] = 1.0

    top3_idx = np.argsort(probs)[::-1][:3]

    results = []
    for rank, idx in enumerate(top3_idx, start=1):
        prob = float(probs[idx])
        results.append({
            "rank":        rank,
            "crop":        _class_names[idx],
            "probability": round(prob, 4),
            "confidence":  _confidence_label(prob),
        })

    return results


# Aliases for flexibility
predict = predict_top3
predict_crop = predict_top3


def _confidence_label(prob: float) -> str:
    if prob >= 0.75:
        return "High"
    elif prob >= 0.50:
        return "Medium"
    else:
        return "Low"


if __name__ == "__main__":
    # Quick smoke test
    result = predict_top3(90, 42, 43, 20.9, 82.0, 6.5, 203.0)
    for r in result:
        print(f"#{r['rank']}: {r['crop']:15s}  {r['probability']:.2%}  [{r['confidence']}]")
