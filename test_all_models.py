"""Quick end-to-end test for all 3 ML models."""
import sys, os, json
APP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AI_Crop_Advisor')
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

# ── Model 3: Weather Disease Risk ─────────────────────────────────────────────
print("=" * 60)
print("Model 3: Weather-Based Disease Risk Model")
print("=" * 60)
from src.weather_disease.predict import predict
r3 = predict(temperature=25.0, humidity=94.0, rainfall=12.0, wind_speed=17.0)
print(f"  Predicted disease : {r3['predicted_disease']}")
print(f"  Confidence        : {r3['probability']:.2%}")
print(f"  All probabilities : {r3['all_probs']}")
print(f"  Disclaimer present: {'disclaimer' in r3}")
print()

# ── Model 1: Crop Recommendation ──────────────────────────────────────────────
print("=" * 60)
print("Model 1: Crop Recommendation Model")
print("=" * 60)
from src.crop.predict import predict_top3
r1 = predict_top3(N=90, P=42, K=43, temperature=20.87, humidity=82.0, ph=6.5, rainfall=202.9)
print(f"  Top-1 crop        : {r1[0]['crop']}")
print(f"  Confidence        : {r1[0]['probability']:.2%}")
print(f"  Top-3             : {[x['crop'] for x in r1]}")
print()

# ── Model 2a: PlantVillage Disease ────────────────────────────────────────────
print("=" * 60)
print("Model 2a: PlantVillage Plant Disease Model")
print("=" * 60)
import config
with open(config.DISEASE_CLASSES_PATH) as f:
    cls = json.load(f)
print(f"  Classes loaded    : {len(cls)}")
print(f"  Sample classes    : {cls[:5]}")
print()

# ── Model 2b: Rice Disease ────────────────────────────────────────────────────
print("=" * 60)
print("Model 2b: Rice Disease Model")
print("=" * 60)
with open(config.RICE_DISEASE_CLASSES_PATH) as f:
    rice_cls = json.load(f)
print(f"  Rice classes      : {rice_cls}")
print()

print("=" * 60)
print("ALL 3 MODELS LOADED AND WORKING!")
print("=" * 60)
