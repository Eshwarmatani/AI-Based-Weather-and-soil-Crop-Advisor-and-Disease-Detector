"""
src/weather_disease/preprocessing.py

Weather-Disease Risk dataset preprocessing (Model 3).

IMPORTANT — Dataset Strategy:
-------------------------------
The downloaded dataset (anshulm257/rice-disease-dataset) contains IMAGES
organised into 6 disease class folders. There is NO tabular weather CSV.

Strategy (documented):
  We generate a scientifically-grounded synthetic weather feature dataset
  based on peer-reviewed literature ranges for rice disease-associated
  weather conditions. Each row represents one historical observation.

  Sources used for feature ranges:
    - Ou (1985) Rice Diseases, CAB International
    - IRRI Rice Disease Management guidelines
    - FAO Agrometeorological guidelines for rice

  Disease classes and representative weather windows:
    ┌───────────────────────────────┬────────────────┬──────────────┬─────────────────┬────────────────┐
    │ Class                         │ Temp (°C)      │ Humidity (%) │ Rainfall (mm/d) │ Wind (km/h)    │
    ├───────────────────────────────┼────────────────┼──────────────┼─────────────────┼────────────────┤
    │ Healthy Rice Leaf             │ 22 – 30        │ 50 – 70      │ 0 – 5           │ 5 – 15         │
    │ Leaf Blast                    │ 20 – 28, peak 25│ 90 – 100    │ 5 – 20          │ 10 – 25        │
    │ Bacterial Leaf Blight         │ 25 – 34        │ 80 – 95      │ 10 – 30         │ 15 – 30        │
    │ Brown Spot                    │ 16 – 36        │ 86 – 100     │ 1 – 10          │ 5 – 20         │
    │ Sheath Blight                 │ 28 – 34        │ 85 – 100     │ 0 – 15          │ 5 – 20         │
    │ Leaf scald                    │ 22 – 32        │ 75 – 95      │ 2 – 15          │ 10 – 25        │
    └───────────────────────────────┴────────────────┴──────────────┴─────────────────┴────────────────┘

  Methodology note (must appear in all outputs):
    "Weather features are generated from literature-documented ranges
     associated with each rice disease class. This model estimates
     weather-based disease risk for rice only and does NOT guarantee
     disease occurrence."

Inputs / Outputs:
  - Generates: data/processed/weather_disease_processed.csv
  - Saves: models/weather_disease_preprocessor.joblib
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

# ─── Literature-based weather feature ranges per disease class ───────────────

DISEASE_WEATHER_PARAMS = {
    "Healthy Rice Leaf": {
        "temp_mean": 26.0, "temp_std": 2.0,
        "humidity_mean": 60.0, "humidity_std": 5.0,
        "rainfall_mean": 2.0,  "rainfall_std": 1.5,
        "wind_mean": 10.0,     "wind_std": 3.0,
        "n_samples": 653,   # matches actual image count per class
    },
    "Leaf Blast": {
        "temp_mean": 24.5, "temp_std": 2.5,
        "humidity_mean": 94.0, "humidity_std": 3.0,
        "rainfall_mean": 12.0, "rainfall_std": 4.0,
        "wind_mean": 17.0, "wind_std": 5.0,
        "n_samples": 634,
    },
    "Bacterial Leaf Blight": {
        "temp_mean": 29.5, "temp_std": 2.5,
        "humidity_mean": 87.0, "humidity_std": 4.0,
        "rainfall_mean": 18.0, "rainfall_std": 6.0,
        "wind_mean": 22.0, "wind_std": 5.0,
        "n_samples": 636,
    },
    "Brown Spot": {
        "temp_mean": 26.0, "temp_std": 4.0,
        "humidity_mean": 92.0, "humidity_std": 4.0,
        "rainfall_mean": 5.0,  "rainfall_std": 2.5,
        "wind_mean": 12.0, "wind_std": 4.0,
        "n_samples": 646,
    },
    "Sheath Blight": {
        "temp_mean": 31.0, "temp_std": 2.0,
        "humidity_mean": 92.0, "humidity_std": 4.0,
        "rainfall_mean": 7.0,  "rainfall_std": 4.0,
        "wind_mean": 12.0, "wind_std": 4.0,
        "n_samples": 632,
    },
    "Leaf scald": {
        "temp_mean": 27.0, "temp_std": 3.0,
        "humidity_mean": 85.0, "humidity_std": 5.0,
        "rainfall_mean": 8.0,  "rainfall_std": 4.0,
        "wind_mean": 17.0, "wind_std": 5.0,
        "n_samples": 628,
    },
}

FEATURES = ["temperature", "humidity", "rainfall", "wind_speed"]
TARGET   = config.WEATHER_TARGET


def generate_dataset(random_state: int = config.RANDOM_STATE) -> pd.DataFrame:
    """
    Generate a synthetic weather-disease dataset from literature-documented
    parameter ranges. Values are clipped to physically plausible limits.
    """
    rng = np.random.default_rng(random_state)
    rows = []

    for disease, params in DISEASE_WEATHER_PARAMS.items():
        n = params["n_samples"]

        temp     = rng.normal(params["temp_mean"],     params["temp_std"],     n)
        humidity = rng.normal(params["humidity_mean"], params["humidity_std"], n)
        rainfall = rng.normal(params["rainfall_mean"], params["rainfall_std"], n)
        wind     = rng.normal(params["wind_mean"],     params["wind_std"],     n)

        # Clip to physical limits
        temp     = np.clip(temp,     5.0, 45.0)
        humidity = np.clip(humidity, 0.0, 100.0)
        rainfall = np.clip(rainfall, 0.0, 100.0)
        wind     = np.clip(wind,     0.0, 60.0)

        for i in range(n):
            rows.append({
                "temperature": round(temp[i], 2),
                "humidity":    round(humidity[i], 2),
                "rainfall":    round(rainfall[i], 2),
                "wind_speed":  round(wind[i], 2),
                TARGET:        disease,
            })

    df = pd.DataFrame(rows).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


def validate(df: pd.DataFrame) -> None:
    """Print validation summary."""
    print("=" * 60)
    print("WEATHER-DISEASE DATASET VALIDATION")
    print("=" * 60)
    print(f"Shape              : {df.shape}")
    print(f"Columns            : {df.columns.tolist()}")
    print(f"Missing values     :\n{df.isnull().sum()}")
    print(f"Duplicate rows     : {df.duplicated().sum()}")
    print(f"\nClass distribution :\n{df[TARGET].value_counts()}")
    print(f"\nStatistical summary:\n{df.describe()}")
    print("=" * 60)
    print("\nMETHODOLOGY NOTE:")
    print("  Weather features are generated from literature-documented ranges")
    print("  associated with each rice disease class. This model estimates")
    print("  weather-based disease risk for rice only and does NOT guarantee")
    print("  disease occurrence.")
    print("=" * 60)


def preprocess(df: pd.DataFrame, save: bool = True):
    """
    Encode labels, split, scale.

    Returns
    -------
    X_train, X_test, y_train, y_test : numpy arrays
    X_train_scaled, X_test_scaled    : scaled arrays
    scaler                           : fitted StandardScaler
    le                               : fitted LabelEncoder
    class_names                      : list of disease strings
    """
    X = df[FEATURES].values
    y_raw = df[TARGET].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = le.classes_.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.WEATHER_TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print(f"Train size   : {X_train.shape[0]} samples")
    print(f"Test size    : {X_test.shape[0]} samples")
    print(f"Classes      : {class_names}")

    if save:
        _save_artifacts(df, scaler, le, class_names,
                        X_train, X_test, y_train, y_test,
                        X_train_scaled, X_test_scaled)

    return (X_train, X_test, y_train, y_test,
            X_train_scaled, X_test_scaled,
            scaler, le, class_names)


def _save_artifacts(df, scaler, le, class_names,
                    X_train, X_test, y_train, y_test,
                    X_train_scaled, X_test_scaled):
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    df.to_csv(config.WEATHER_PROCESSED_PATH, index=False)
    joblib.dump(scaler, config.WEATHER_PREPROCESSOR_PATH)

    classes_path = os.path.join(config.MODELS_DIR, "weather_disease_classes.json")
    with open(classes_path, "w") as f:
        json.dump(class_names, f)

    np_dir = os.path.join(config.PROCESSED_DIR, "weather_splits")
    os.makedirs(np_dir, exist_ok=True)
    np.save(os.path.join(np_dir, "X_train.npy"), X_train)
    np.save(os.path.join(np_dir, "X_test.npy"), X_test)
    np.save(os.path.join(np_dir, "y_train.npy"), y_train)
    np.save(os.path.join(np_dir, "y_test.npy"), y_test)
    np.save(os.path.join(np_dir, "X_train_scaled.npy"), X_train_scaled)
    np.save(os.path.join(np_dir, "X_test_scaled.npy"), X_test_scaled)

    print(f"Preprocessor saved  : {config.WEATHER_PREPROCESSOR_PATH}")
    print(f"Class names saved   : {classes_path}")
    print(f"Processed CSV saved : {config.WEATHER_PROCESSED_PATH}")


if __name__ == "__main__":
    df = generate_dataset()
    validate(df)
    preprocess(df, save=True)
    print("\nWeather-disease preprocessing complete.")
