"""
src/disease/predict_rice.py
Inference module for Rice Leaf Disease detection using the trained rice MobileNetV2 model.
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

_rice_model = None
_rice_class_names = None


def _load_rice():
    global _rice_model, _rice_class_names
    if _rice_model is None:
        if not os.path.exists(config.RICE_DISEASE_MODEL_PATH):
            raise FileNotFoundError(
                f"Rice disease model not found at {config.RICE_DISEASE_MODEL_PATH}. "
                "Run src/disease/train_rice.py first."
            )
        import tensorflow as tf
        _rice_model = tf.keras.models.load_model(config.RICE_DISEASE_MODEL_PATH)

    if _rice_class_names is None:
        with open(config.RICE_DISEASE_CLASSES_PATH) as f:
            _rice_class_names = json.load(f)


def predict_rice_from_path(image_path: str) -> dict:
    """Predict rice disease class from an image file path."""
    import tensorflow as tf

    _load_rice()

    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, config.IMAGE_SIZE)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = tf.expand_dims(img, axis=0)

    probs = _rice_model.predict(img, verbose=0)[0]
    top3_idx = np.argsort(probs)[::-1][:3]

    top3 = [
        {"class": _rice_class_names[i], "confidence": round(float(probs[i]), 4)}
        for i in top3_idx
    ]

    return {
        "predicted_class": _rice_class_names[top3_idx[0]],
        "confidence":      round(float(probs[top3_idx[0]]), 4),
        "low_confidence":  float(probs[top3_idx[0]]) < config.CONFIDENCE_THRESHOLD,
        "top3":            top3,
    }


def predict_rice_from_array(img_array: np.ndarray) -> dict:
    """Predict rice disease from a numpy array (H x W x 3)."""
    import tensorflow as tf

    _load_rice()

    img = tf.image.resize(img_array, config.IMAGE_SIZE)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(tf.cast(img, tf.float32))
    img = tf.expand_dims(img, axis=0)

    probs = _rice_model.predict(img, verbose=0)[0]
    top3_idx = np.argsort(probs)[::-1][:3]

    top3 = [
        {"class": _rice_class_names[i], "confidence": round(float(probs[i]), 4)}
        for i in top3_idx
    ]

    return {
        "predicted_class": _rice_class_names[top3_idx[0]],
        "confidence":      round(float(probs[top3_idx[0]]), 4),
        "low_confidence":  float(probs[top3_idx[0]]) < config.CONFIDENCE_THRESHOLD,
        "top3":            top3,
    }
