"""
src/disease/predict.py

Plant Disease Detection inference module.
Loads the saved MobileNetV2 model and predicts the disease from an image.
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

_model       = None
_class_names = None


def _load():
    global _model, _class_names

    if _model is None:
        if not os.path.exists(config.DISEASE_MODEL_PATH):
            raise FileNotFoundError(
                f"Disease model not found at {config.DISEASE_MODEL_PATH}. "
                "Run src/disease/train.py first."
            )
        import tensorflow as tf
        _model = tf.keras.models.load_model(config.DISEASE_MODEL_PATH)

    if _class_names is None:
        with open(config.DISEASE_CLASSES_PATH) as f:
            _class_names = json.load(f)


def predict_from_path(image_path: str) -> dict:
    """
    Predict disease class from an image file path.

    Returns
    -------
    dict:
        predicted_class : str  (e.g. "Tomato_Early_blight")
        confidence      : float (0–1)
        low_confidence  : bool
        top3            : list of {class, confidence}
    """
    import tensorflow as tf

    _load()

    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, config.IMAGE_SIZE)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = tf.expand_dims(img, axis=0)

    probs = _model.predict(img, verbose=0)[0]
    top3_idx = np.argsort(probs)[::-1][:3]

    top3 = [
        {"class": _class_names[i], "confidence": round(float(probs[i]), 4)}
        for i in top3_idx
    ]

    return {
        "predicted_class": _class_names[top3_idx[0]],
        "confidence":      round(float(probs[top3_idx[0]]), 4),
        "low_confidence":  float(probs[top3_idx[0]]) < config.CONFIDENCE_THRESHOLD,
        "top3":            top3,
    }


def predict_from_array(img_array: np.ndarray) -> dict:
    """
    Predict from a pre-loaded numpy array (H x W x 3, uint8 or float).
    Convenience wrapper for Flask uploads.
    """
    import tensorflow as tf

    _load()

    img = tf.image.resize(img_array, config.IMAGE_SIZE)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(tf.cast(img, tf.float32))
    img = tf.expand_dims(img, axis=0)

    probs = _model.predict(img, verbose=0)[0]
    top3_idx = np.argsort(probs)[::-1][:3]

    top3 = [
        {"class": _class_names[i], "confidence": round(float(probs[i]), 4)}
        for i in top3_idx
    ]

    return {
        "predicted_class": _class_names[top3_idx[0]],
        "confidence":      round(float(probs[top3_idx[0]]), 4),
        "low_confidence":  float(probs[top3_idx[0]]) < config.CONFIDENCE_THRESHOLD,
        "top3":            top3,
    }
