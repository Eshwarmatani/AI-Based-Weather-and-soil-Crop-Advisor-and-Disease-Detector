"""
src/dataset_inspection.py

Phase 2 — Dataset inspection and EDA summary.
Run this script to validate all three datasets and generate summary reports.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

def check_datasets():
    """Check all three datasets exist and print a summary."""
    print("=" * 70)
    print("PHASE 2 - DATASET INSPECTION REPORT")
    print("=" * 70)

    all_ok = True

    # ── Dataset 1: Crop Recommendation ──────────────────────────────────────
    print("\n[1] CROP RECOMMENDATION DATASET")
    print("-" * 40)
    if not os.path.exists(config.CROP_CSV_PATH):
        print(f"  MISSING: {config.CROP_CSV_PATH}")
        all_ok = False
    else:
        import pandas as pd
        df = pd.read_csv(config.CROP_CSV_PATH)
        print(f"  Path     : {config.CROP_CSV_PATH}")
        print(f"  Shape    : {df.shape}")
        print(f"  Columns  : {df.columns.tolist()}")
        print(f"  Missing  : {df.isnull().sum().sum()} values")
        print(f"  Dupes    : {df.duplicated().sum()} rows")
        print(f"  Classes  : {df['label'].nunique()} crops")
        print("  Min/Max class sizes:")
        vc = df['label'].value_counts()
        print(f"    Min: {vc.min()} ({vc.idxmin()})")
        print(f"    Max: {vc.max()} ({vc.idxmax()})")
        print("  Status   : OK")

    # ── Dataset 2: Plant Disease ─────────────────────────────────────────────
    print("\n[2] PLANT DISEASE DATASET (PlantVillage)")
    print("-" * 40)
    if not os.path.exists(config.PLANT_DISEASE_DIR):
        print(f"  MISSING: {config.PLANT_DISEASE_DIR}")
        all_ok = False
    else:
        all_folders = sorted(os.listdir(config.PLANT_DISEASE_DIR))
        valid_classes = []
        total_images = 0
        print(f"  Path     : {config.PLANT_DISEASE_DIR}")
        print(f"  All folders found: {len(all_folders)}")
        for cls in all_folders:
            if cls in config.PLANT_DISEASE_EXCLUDE:
                print(f"  [EXCLUDED] {cls}")
                continue
            p = os.path.join(config.PLANT_DISEASE_DIR, cls)
            if os.path.isdir(p):
                n = len([f for f in os.listdir(p) if f.lower().endswith(('.jpg','.jpeg','.png'))])
                if n > 0:
                    valid_classes.append(cls)
                    total_images += n
        print(f"  Valid classes : {len(valid_classes)}")
        print(f"  Total images  : {total_images}")
        print("  Status        : OK")

    # ── Dataset 3: Rice Disease ──────────────────────────────────────────────
    print("\n[3] RICE DISEASE DATASET")
    print("-" * 40)
    if not os.path.exists(config.RICE_DISEASE_DIR):
        print(f"  MISSING: {config.RICE_DISEASE_DIR}")
        all_ok = False
    else:
        classes = sorted(os.listdir(config.RICE_DISEASE_DIR))
        total = 0
        print(f"  Path     : {config.RICE_DISEASE_DIR}")
        for cls in classes:
            p = os.path.join(config.RICE_DISEASE_DIR, cls)
            n = len([f for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))])
            print(f"  {cls:30s}: {n} images")
            total += n
        print(f"  Total images : {total}")
        print("  Note: Image-based dataset. Weather features will be")
        print("        generated from literature-documented ranges.")
        print("  Status       : OK")

    print("\n" + "=" * 70)
    if all_ok:
        print("ALL DATASETS PRESENT AND VALID. Ready for Phase 3.")
    else:
        print("SOME DATASETS ARE MISSING. Please run download_datasets.py.")
    print("=" * 70)
    return all_ok


if __name__ == "__main__":
    try:
        import pandas  # noqa: F401
        check_datasets()
    except ImportError:
        print("ERROR: pandas not installed. Run: pip install -r requirements.txt")
        sys.exit(1)
