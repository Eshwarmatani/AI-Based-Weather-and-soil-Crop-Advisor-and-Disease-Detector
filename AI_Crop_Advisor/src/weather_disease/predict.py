"""
src/weather_disease/predict.py

Weather-Based Disease Risk inference module.
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

DISCLAIMER = (
    "Weather features are generated from literature-documented ranges "
    "associated with rice disease classes. This model estimates "
    "weather-based disease risk for rice only and does NOT guarantee "
    "disease occurrence."
)


def _load():
    global _model, _class_names

    if _model is None:
        if not os.path.exists(config.WEATHER_MODEL_PATH):
            raise FileNotFoundError(
                f"Weather disease model not found at {config.WEATHER_MODEL_PATH}. "
                "Run src/weather_disease/train.py first."
            )
        _model = joblib.load(config.WEATHER_MODEL_PATH)

    if _class_names is None:
        path = os.path.join(config.MODELS_DIR, "weather_disease_classes.json")
        with open(path) as f:
            _class_names = json.load(f)


def predict(temperature: float, humidity: float,
            rainfall: float, wind_speed: float) -> dict:
    """
    Predict disease risk from weather conditions.

    Returns
    -------
    dict with keys:
        predicted_disease : str
        probability       : float
        all_probs         : dict {class_name: prob}
        disclaimer        : str
    """
    _load()

    features = np.array([[temperature, humidity, rainfall, wind_speed]])

    if hasattr(_model, "predict_proba"):
        probs     = _model.predict_proba(features)[0]
        pred_idx  = int(np.argmax(probs))
        pred_prob = float(probs[pred_idx])
        all_probs = {name: round(float(p), 4)
                     for name, p in zip(_class_names, probs)}
    else:
        pred_idx  = int(_model.predict(features)[0])
        pred_prob = 1.0
        all_probs = {name: (1.0 if i == pred_idx else 0.0)
                     for i, name in enumerate(_class_names)}

    return {
        "predicted_disease": _class_names[pred_idx],
        "probability":       round(pred_prob, 4),
        "all_probs":         all_probs,
        "disclaimer":        DISCLAIMER,
    }


if __name__ == "__main__":
    r = predict(temperature=25.0, humidity=94.0, rainfall=12.0, wind_speed=17.0)
    print(f"Predicted: {r['predicted_disease']}  ({r['probability']:.2%})")
    print(f"All probs: {r['all_probs']}")
