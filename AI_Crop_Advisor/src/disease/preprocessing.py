"""
src/disease/preprocessing.py

Preprocessing for the Plant Disease Image dataset (Model 2 — MobileNetV2).

Responsibilities:
  - Locate the PlantVillage directory
  - Detect available class folders dynamically (no hard-coded class list)
  - Exclude non-disease folders (e.g. a nested 'PlantVillage' directory)
  - Gather image paths and labels
  - Perform stratified train / validation / test split
  - Save class names to models/disease_classes.json
  - Return tf.data pipelines for training

IMPORTANT: Data augmentation is applied ONLY to the training pipeline.
           Validation and test pipelines use only normalisation.
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def discover_classes(base_dir: str | None = None) -> list[str]:
    """
    Dynamically detect valid disease class folders.
    Excludes any folder listed in config.PLANT_DISEASE_EXCLUDE.
    Returns sorted list of class names.
    """
    root = base_dir or config.PLANT_DISEASE_DIR

    if not os.path.exists(root):
        raise FileNotFoundError(
            f"PlantVillage directory not found at: {root}\n"
            "Please run download_datasets.py first."
        )

    classes = []
    for name in sorted(os.listdir(root)):
        if name in config.PLANT_DISEASE_EXCLUDE:
            print(f"  [SKIP] Excluded folder: {name}")
            continue
        full = os.path.join(root, name)
        if os.path.isdir(full):
            n_images = len([f for f in os.listdir(full)
                            if f.lower().endswith((".jpg", ".jpeg", ".png"))])
            if n_images > 0:
                classes.append(name)
                print(f"  [OK]   {name}: {n_images} images")
            else:
                print(f"  [SKIP] {name}: 0 images - skipping empty folder")

    print(f"\nTotal valid classes: {len(classes)}")
    return classes


def collect_image_paths(classes: list[str], base_dir: str | None = None):
    """
    Walk each class folder and return parallel lists of
    (image_path, class_index, class_name).
    """
    root = base_dir or config.PLANT_DISEASE_DIR
    paths, labels, names = [], [], []

    for idx, cls in enumerate(classes):
        folder = os.path.join(root, cls)
        for fname in os.listdir(folder):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(folder, fname))
                labels.append(idx)
                names.append(cls)

    return paths, labels, names


def stratified_split(paths, labels, random_state: int = config.RANDOM_STATE):
    """
    Stratified train / val / test split.
    Ratios: 70% train, 15% val, 15% test.
    """
    from sklearn.model_selection import train_test_split

    paths  = np.array(paths)
    labels = np.array(labels)

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        paths, labels,
        test_size=(config.VAL_SPLIT + config.TEST_SPLIT),
        random_state=random_state,
        stratify=labels,
    )

    # Split the remaining 30% into 15% val + 15% test
    val_ratio = config.VAL_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=(1.0 - val_ratio),
        random_state=random_state,
        stratify=y_tmp,
    )

    print(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_class_names(classes: list[str]) -> None:
    """Save class names list to models/disease_classes.json."""
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    with open(config.DISEASE_CLASSES_PATH, "w") as f:
        json.dump(classes, f, indent=2)
    print(f"Disease class names saved: {config.DISEASE_CLASSES_PATH}")


def load_class_names() -> list[str]:
    """Load class names from models/disease_classes.json."""
    with open(config.DISEASE_CLASSES_PATH, "r") as f:
        return json.load(f)


def build_tf_datasets(X_train, X_val, X_test, y_train, y_val, y_test,
                      n_classes: int, batch_size: int = config.BATCH_SIZE):
    """
    Build tf.data.Dataset objects for train, val, test.
    Augmentation is ONLY applied to the training set.

    Returns (train_ds, val_ds, test_ds) ready for model.fit().
    """
    import tensorflow as tf

    def _load_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, config.IMAGE_SIZE)
        # MobileNetV2 expects [-1, 1] via preprocess_input
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        return img, label

    def _augment(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, max_delta=0.2)
        img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
        return img, label

    AUTOTUNE = tf.data.AUTOTUNE

    # Training pipeline — with augmentation
    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(len(X_train), seed=config.RANDOM_STATE)
        .map(_load_image, num_parallel_calls=AUTOTUNE)
        .map(_augment,     num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    # Validation pipeline — NO augmentation
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .map(_load_image, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    # Test pipeline — NO augmentation
    test_ds = (
        tf.data.Dataset.from_tensor_slices((X_test, y_test))
        .map(_load_image, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    return train_ds, val_ds, test_ds


def run_preprocessing(base_dir: str | None = None, save: bool = True):
    """
    Full preprocessing pipeline for the disease image model.
    Returns split arrays and class names.
    """
    print("=" * 60)
    print("PLANT DISEASE DATASET VALIDATION")
    print("=" * 60)

    classes = discover_classes(base_dir)

    if save:
        save_class_names(classes)

    paths, labels, _ = collect_image_paths(classes, base_dir)
    print(f"\nTotal images collected : {len(paths)}")

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(paths, labels)

    print("=" * 60)
    return X_train, X_val, X_test, y_train, y_val, y_test, classes


if __name__ == "__main__":
    run_preprocessing(save=True)
    print("\nDisease image preprocessing complete.")
