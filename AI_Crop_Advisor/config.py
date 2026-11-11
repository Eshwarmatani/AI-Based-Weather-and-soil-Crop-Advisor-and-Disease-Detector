"""
config.py — Central project configuration.
All paths, constants, and model parameters are defined here.
"""
import os

# ─── Base Paths ───────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Environment Loader ────────────────────────────────────────────────────────
def _load_dotenv():
    """Load variables from .env if present without requiring external package."""
    candidates = [
        os.path.join(BASE_DIR, ".env"),
        os.path.join(os.path.dirname(BASE_DIR), ".env")
    ]
    for env_path in candidates:
        if os.path.isfile(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip("'\"")
                            if k and k not in os.environ:
                                os.environ[k] = v
            except Exception:
                pass

_load_dotenv()

# ─── Chatbot Configuration ───────────────────────────────────────────────────
GEMINI_API_KEY   = os.environ.get("GEMINI_API_KEY", "").strip()
OPENAI_API_KEY   = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL  = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
OPENAI_MODEL     = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip()
CHATBOT_PROVIDER = os.environ.get("CHATBOT_PROVIDER", "auto").strip().lower()  # auto, gemini, openai, local
CHATBOT_MODEL    = os.environ.get("CHATBOT_MODEL", "gemini-1.5-flash").strip()


DATA_DIR           = os.path.join(BASE_DIR, "data")
MODELS_DIR         = os.path.join(BASE_DIR, "models")
REPORTS_DIR        = os.path.join(BASE_DIR, "reports")
FIGURES_DIR        = os.path.join(REPORTS_DIR, "figures")
RESULTS_DIR        = os.path.join(REPORTS_DIR, "model_results")
UPLOADS_DIR        = os.path.join(BASE_DIR, "uploads")
DATABASE_DIR       = os.path.join(BASE_DIR, "database")

# ─── Kaggle Cache Paths ───────────────────────────────────────────────────────

KAGGLE_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "kagglehub", "datasets")

CROP_CSV_PATH = os.path.join(
    KAGGLE_CACHE,
    "atharvaingle", "crop-recommendation-dataset", "versions", "1",
    "Crop_recommendation.csv"
)

PLANT_DISEASE_DIR = os.path.join(
    KAGGLE_CACHE,
    "emmarex", "plantdisease", "versions", "1", "PlantVillage"
)

RICE_DISEASE_DIR = os.path.join(
    KAGGLE_CACHE,
    "anshulm257", "rice-disease-dataset", "versions", "1", "Rice_Leaf_AUG"
)

# ─── Model Save Paths ─────────────────────────────────────────────────────────

CROP_MODEL_PATH        = os.path.join(MODELS_DIR, "crop_model.joblib")
CROP_PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "crop_preprocessor.joblib")
CROP_CLASSES_PATH      = os.path.join(MODELS_DIR, "crop_classes.json")
WEATHER_CLASSES_PATH   = os.path.join(MODELS_DIR, "weather_disease_classes.json")

DISEASE_MODEL_PATH     = os.path.join(MODELS_DIR, "disease_model.keras")
DISEASE_CLASSES_PATH   = os.path.join(MODELS_DIR, "disease_classes.json")

RICE_DISEASE_MODEL_PATH   = os.path.join(MODELS_DIR, "rice_disease_model.keras")
RICE_DISEASE_CLASSES_PATH = os.path.join(MODELS_DIR, "rice_disease_classes.json")

WEATHER_MODEL_PATH        = os.path.join(MODELS_DIR, "weather_disease_model.joblib")
WEATHER_PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "weather_disease_preprocessor.joblib")

# ─── Processed Data Paths ─────────────────────────────────────────────────────

PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
CROP_PROCESSED_PATH    = os.path.join(PROCESSED_DIR, "crop_processed.csv")
WEATHER_PROCESSED_PATH = os.path.join(PROCESSED_DIR, "weather_disease_processed.csv")

# ─── Model 1 — Crop Recommendation ───────────────────────────────────────────

CROP_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
CROP_TARGET   = "label"
CROP_TEST_SIZE = 0.20
RANDOM_STATE   = 42

# ─── Model 2 — Plant Disease Detection ───────────────────────────────────────

IMAGE_SIZE        = (224, 224)
BATCH_SIZE        = 32
EPOCHS            = 30  # EarlyStopping will cut this short
LEARNING_RATE     = 1e-4
TRAIN_SPLIT       = 0.70
VAL_SPLIT         = 0.15
TEST_SPLIT        = 0.15
CONFIDENCE_THRESHOLD = 0.60   # Below this -> show "low confidence" warning

# Classes to exclude from PlantVillage (folder names that are not leaf disease classes)
PLANT_DISEASE_EXCLUDE = ["PlantVillage"]

# ─── Model 3 — Weather Disease Risk ──────────────────────────────────────────

WEATHER_TEST_SIZE = 0.20
WEATHER_TARGET = "disease_label"
WEATHER_DISEASE_FEATURES = ["temperature", "humidity", "rainfall", "wind_speed"]

# ─── Static Figures (served by Flask from static/figures) ─────────────────────
STATIC_FIGURES_DIR = os.path.join(BASE_DIR, "app", "static", "figures")

# ─── Database ─────────────────────────────────────────────────────────────────

DATABASE_PATH = os.path.join(DATABASE_DIR, "agriculture.db")
DATABASE_URI  = f"sqlite:///{DATABASE_PATH}"

# ─── Flask Upload ──────────────────────────────────────────────────────────────

ALLOWED_EXTENSIONS  = {"jpg", "jpeg", "png"}
MAX_UPLOAD_SIZE_MB  = 10


if __name__ == "__main__":
    print("=" * 65)
    print("  AI-BASED CROP ADVISOR - SYSTEM CONFIGURATION CHECK")
    print("=" * 65)
    print(f"Base Directory       : {BASE_DIR}")
    print(f"Database Path        : {DATABASE_PATH}")
    print(f"  [DB Exists]        : {os.path.exists(DATABASE_PATH)}")
    print("-" * 65)
    print("Datasets:")
    print(f"  Crop Dataset       : {os.path.exists(CROP_CSV_PATH)} ({CROP_CSV_PATH})")
    print(f"  PlantVillage       : {os.path.exists(PLANT_DISEASE_DIR)} ({PLANT_DISEASE_DIR})")
    print(f"  Rice Dataset       : {os.path.exists(RICE_DISEASE_DIR)} ({RICE_DISEASE_DIR})")
    print("-" * 65)
    print("Trained Models:")
    print(f"  Crop Model         : {'READY' if os.path.exists(CROP_MODEL_PATH) else 'NOT FOUND'} ({CROP_MODEL_PATH})")
    print(f"  Weather Risk Model : {'READY' if os.path.exists(WEATHER_MODEL_PATH) else 'NOT FOUND'} ({WEATHER_MODEL_PATH})")
    print(f"  Disease CNN Model  : {'READY' if os.path.exists(DISEASE_MODEL_PATH) else 'TRAINING / NOT SAVED YET'} ({DISEASE_MODEL_PATH})")
    print("=" * 65)
    print("To launch the Flask Web Server, run: python AI_Crop_Advisor/run.py")
    print("Server URL: http://localhost:5000")
    print("=" * 65)

