"""
src/crop/preprocessing.py
Preprocessing pipeline for the Crop Recommendation dataset (Model 1).

Responsibilities:
  - Load raw CSV
  - Validate columns and data types
  - Check for missing values and duplicates
  - Encode target labels
  - Split into train / test
  - Fit and return a StandardScaler (for algorithms that require scaling)
  - Save processed data to data/processed/
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def load_and_validate(csv_path: str | None = None) -> pd.DataFrame:
    """Load the crop CSV and run validation checks."""
    path = csv_path or config.CROP_CSV_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Crop dataset not found at: {path}\n"
            "Please run download_datasets.py first."
        )

    df = pd.read_csv(path)

    # ── Validation ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("CROP DATASET VALIDATION")
    print("=" * 60)
    print(f"Shape         : {df.shape}")
    print(f"\nColumns       : {df.columns.tolist()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    print(f"\nClass distribution:\n{df[config.CROP_TARGET].value_counts()}")
    print(f"\nStatistical summary:\n{df.describe()}")
    print("=" * 60)

    expected_cols = config.CROP_FEATURES + [config.CROP_TARGET]
    missing_cols  = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    return df


def preprocess(df: pd.DataFrame, save: bool = True):
    """
    Preprocess crop data.

    Returns
    -------
    X_train, X_test, y_train, y_test : raw (unscaled) numpy arrays
    X_train_scaled, X_test_scaled    : scaled arrays for distance-based models
    scaler                           : fitted StandardScaler
    label_encoder                    : fitted LabelEncoder
    class_names                      : list of crop label strings
    """
    # Drop duplicates (if any)
    df = df.drop_duplicates().copy()

    X = df[config.CROP_FEATURES].values
    y_raw = df[config.CROP_TARGET].values

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = le.classes_.tolist()

    # Train / test split (stratified to preserve class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.CROP_TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )

    # Fit scaler on TRAIN only → transform both
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print(f"Train size : {X_train.shape[0]} samples")
    print(f"Test size  : {X_test.shape[0]} samples")
    print(f"Classes    : {class_names}")

    if save:
        _save_artifacts(X_train, X_test, y_train, y_test,
                        X_train_scaled, X_test_scaled,
                        scaler, le, class_names, df)

    return (X_train, X_test, y_train, y_test,
            X_train_scaled, X_test_scaled,
            scaler, le, class_names)


def _save_artifacts(X_train, X_test, y_train, y_test,
                    X_train_scaled, X_test_scaled,
                    scaler, le, class_names, df_clean):
    """Persist preprocessing artifacts."""
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    # Save processed CSV
    df_clean.to_csv(config.CROP_PROCESSED_PATH, index=False)

    # Save scaler and encoder
    joblib.dump(scaler, config.CROP_PREPROCESSOR_PATH)

    # Save class names as JSON for the Flask app
    classes_path = os.path.join(config.MODELS_DIR, "crop_classes.json")
    with open(classes_path, "w") as f:
        json.dump(class_names, f)

    # Save split arrays for reproducibility
    np_dir = os.path.join(config.PROCESSED_DIR, "crop_splits")
    os.makedirs(np_dir, exist_ok=True)
    np.save(os.path.join(np_dir, "X_train.npy"), X_train)
    np.save(os.path.join(np_dir, "X_test.npy"), X_test)
    np.save(os.path.join(np_dir, "y_train.npy"), y_train)
    np.save(os.path.join(np_dir, "y_test.npy"), y_test)
    np.save(os.path.join(np_dir, "X_train_scaled.npy"), X_train_scaled)
    np.save(os.path.join(np_dir, "X_test_scaled.npy"), X_test_scaled)

    print(f"Preprocessor saved  : {config.CROP_PREPROCESSOR_PATH}")
    print(f"Class names saved   : {classes_path}")
    print(f"Processed CSV saved : {config.CROP_PROCESSED_PATH}")


if __name__ == "__main__":
    df = load_and_validate()
    preprocess(df, save=True)
    print("\nCrop preprocessing complete.")
