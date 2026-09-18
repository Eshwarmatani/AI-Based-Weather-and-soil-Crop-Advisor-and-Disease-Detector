"""
src/explainability/shap_explainer.py

SHAP explainability for the Crop Recommendation and Weather Risk tabular models.

IMPORTANT:
  - SHAP explanations are computed ONLY for tabular models (Model 1 and Model 3).
  - MobileNetV2 (Model 2) is explained via Class Activation Maps, not SHAP.
  - SHAP values quantify feature contributions; positive = increases probability,
    negative = decreases probability.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

# Lazy imports to keep startup fast
_crop_explainer    = None
_weather_explainer = None


def _get_crop_explainer():
    """Build and cache SHAP TreeExplainer for the crop model."""
    global _crop_explainer
    if _crop_explainer is None:
        import shap
        import joblib
        model = joblib.load(config.CROP_MODEL_PATH)
        # For Pipeline, extract the actual estimator if present
        est = getattr(model, "named_steps", {}).get("clf", model)
        _crop_explainer = shap.TreeExplainer(est)
    return _crop_explainer


def _get_weather_explainer():
    """Build and cache SHAP explainer for the weather risk model."""
    global _weather_explainer
    if _weather_explainer is None:
        import shap
        import joblib
        model = joblib.load(config.WEATHER_MODEL_PATH)
        est = getattr(model, "named_steps", {}).get("clf", model)
        # SVM -> use KernelExplainer with small background sample
        if hasattr(est, "kernel"):
            bg = np.zeros((1, 4))  # neutral background
            _weather_explainer = shap.KernelExplainer(model.predict_proba, bg)
        else:
            _weather_explainer = shap.TreeExplainer(est)
    return _weather_explainer


def explain_crop(N: float, P: float, K: float,
                 temperature: float, humidity: float,
                 ph: float, rainfall: float) -> dict:
    """
    Return SHAP feature importances for a crop prediction.

    Returns
    -------
    dict:
        feature_names : list of str
        shap_values   : list of float (for predicted class)
        base_value    : float
        predicted_class_idx : int
        note          : str
    """
    try:
        import joblib
        import pandas as pd
        model = joblib.load(config.CROP_MODEL_PATH)
        features_df = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=config.CROP_FEATURES)
        pred_class = int(model.predict(features_df)[0])

        explainer  = _get_crop_explainer()
        # For Pipeline scaler-aware, transform if needed
        if hasattr(model, "named_steps") and "scaler" in model.named_steps:
            features_in = model.named_steps["scaler"].transform(features_df)
        else:
            features_in = features_df.values

        sv = explainer.shap_values(features_in)

        # sv shape: (n_classes, n_samples, n_features) or (n_samples, n_features, n_classes)
        if isinstance(sv, list):
            class_sv = sv[pred_class][0]
            exp_val = explainer.expected_value[pred_class] if hasattr(explainer.expected_value, "__getitem__") else explainer.expected_value
            base_val = float(exp_val)
        elif getattr(sv, "ndim", 0) == 3:
            class_sv = sv[0, :, pred_class]
            exp_val = explainer.expected_value[pred_class] if hasattr(explainer.expected_value, "__getitem__") else explainer.expected_value
            base_val = float(exp_val)
        else:
            class_sv = sv[0]
            exp_val = explainer.expected_value[pred_class] if hasattr(explainer.expected_value, "__getitem__") else explainer.expected_value
            base_val = float(exp_val)

        return {
            "feature_names":      config.CROP_FEATURES,
            "shap_values":        [round(float(v), 5) for v in class_sv],
            "base_value":         round(base_val, 5),
            "predicted_class_idx": pred_class,
            "note": (
                "SHAP values show each feature's contribution to the predicted crop. "
                "Positive = supports this crop, Negative = works against it."
            ),
        }
    except Exception as e:
        return {"error": str(e), "note": "SHAP explanation unavailable."}


def explain_crop_dict(inputs: dict) -> dict:
    """Wrapper that accepts a dict of feature values."""
    return explain_crop(**{k: float(v) for k, v in inputs.items()
                           if k in config.CROP_FEATURES + ["N", "P", "K",
                           "temperature", "humidity", "ph", "rainfall"]})


def explain_weather(temperature: float, humidity: float, rainfall: float, wind_speed: float) -> dict:
    """
    Return SHAP feature importances for a weather risk prediction.
    """
    try:
        import joblib
        model = joblib.load(config.WEATHER_MODEL_PATH)
        features = np.array([[temperature, humidity, rainfall, wind_speed]])
        pred_class = int(model.predict(features)[0])

        explainer = _get_weather_explainer()
        # If KernelExplainer was built on model.predict_proba pipeline, pass raw features
        # If TreeExplainer was built on underlying clf, scale if model has scaler
        import shap
        if isinstance(explainer, shap.KernelExplainer):
            features_in = features
        elif hasattr(model, "named_steps") and "scaler" in model.named_steps:
            features_in = model.named_steps["scaler"].transform(features)
        else:
            features_in = features

        sv = explainer.shap_values(features_in)
        
        if isinstance(sv, list):
            class_sv = sv[pred_class][0]
            exp_val = explainer.expected_value[pred_class] if hasattr(explainer.expected_value, "__getitem__") else explainer.expected_value
            base_val = float(exp_val)
        elif getattr(sv, "ndim", 0) == 3:
            class_sv = sv[0, :, pred_class]
            exp_val = explainer.expected_value[pred_class] if hasattr(explainer.expected_value, "__getitem__") else explainer.expected_value
            base_val = float(exp_val)
        else:
            class_sv = sv[0]
            exp_val = explainer.expected_value[pred_class] if hasattr(explainer.expected_value, "__getitem__") else explainer.expected_value
            base_val = float(exp_val)
            
        return {
            "feature_names": config.WEATHER_DISEASE_FEATURES,
            "shap_values": [round(float(v), 5) for v in class_sv],
            "base_value": round(base_val, 5),
            "predicted_class_idx": pred_class,
            "note": "SHAP values show feature contributions to disease risk."
        }
    except Exception as e:
        return {"error": str(e), "note": "SHAP explanation unavailable."}


def explain_weather_dict(inputs: dict) -> dict:
    return explain_weather(**{k: float(v) for k, v in inputs.items()
                              if k in config.WEATHER_DISEASE_FEATURES})
