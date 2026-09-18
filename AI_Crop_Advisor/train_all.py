"""
train_all.py — Master training script.

Trains all three models sequentially:
  Phase 2 — Dataset inspection
  Phase 3 — Crop Recommendation model
  Phase 4 — Plant Disease Detection model (MobileNetV2)
  Phase 5 — Weather Disease Risk model

Usage:
    python AI_Crop_Advisor/train_all.py

  OR to skip Phase 4 (if TensorFlow is not installed):
    python AI_Crop_Advisor/train_all.py --skip-disease
"""

import sys
import os
import argparse
import shutil

# Ensure project root is on path
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import config


def copy_figures():
    """Copy report figures to static/figures for Flask serving."""
    src = config.FIGURES_DIR
    dst = config.STATIC_FIGURES_DIR
    os.makedirs(dst, exist_ok=True)
    if os.path.exists(src):
        for f in os.listdir(src):
            if f.endswith(".png"):
                shutil.copy(os.path.join(src, f), os.path.join(dst, f))
        print(f"Figures copied to: {dst}")


def main():
    parser = argparse.ArgumentParser(description="Train all AI Crop Advisor models")
    parser.add_argument("--skip-disease", action="store_true",
                        help="Skip Phase 4 (plant disease MobileNetV2) if TF not installed")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  AI CROP ADVISOR - MASTER TRAINING SCRIPT")
    print("=" * 70)

    # ── Phase 2: Dataset Inspection ───────────────────────────────────────────
    print("\n>>> PHASE 2: Dataset Inspection\n")
    from src.dataset_inspection import check_datasets
    if not check_datasets():
        print("\nERROR: Not all datasets found. Please run download_datasets.py first.")
        sys.exit(1)

    # ── Phase 3: Crop Model ───────────────────────────────────────────────────
    print("\n>>> PHASE 3: Crop Recommendation Model\n")
    try:
        from src.crop.train import main as train_crop
    except ImportError:
        from src.crop.train import train_and_evaluate_all as train_crop
    train_crop()
    copy_figures()

    # ── Phase 4: Disease Image Model ──────────────────────────────────────────
    if not args.skip_disease:
        print("\n>>> PHASE 4: Plant Disease Detection Model\n")
        try:
            import tensorflow as tf
            from src.disease.train import train as train_disease
            train_disease()
            copy_figures()
        except ImportError:
            print("[SKIP] TensorFlow not installed. Run:")
            print("       pip install tensorflow-cpu")
            print("       Then re-run: python train_all.py")
    else:
        print("\n>>> PHASE 4: [SKIPPED - --skip-disease flag set]\n")

    # ── Phase 5: Weather Risk Model ───────────────────────────────────────────
    print("\n>>> PHASE 5: Weather Disease Risk Model\n")
    from src.weather_disease.train import main as train_weather
    train_weather()
    copy_figures()

    print("\n" + "=" * 70)
    print("  ALL TRAINING COMPLETE")
    print("=" * 70)
    print("\nModels saved:")
    for p in [config.CROP_MODEL_PATH, config.DISEASE_MODEL_PATH, config.WEATHER_MODEL_PATH]:
        exists = "OK" if os.path.exists(p) else "MISSING"
        print(f"  [{exists}] {os.path.basename(p)}")

    print("\nStart the web app:")
    print("  python AI_Crop_Advisor/run.py")
    print()


if __name__ == "__main__":
    main()
