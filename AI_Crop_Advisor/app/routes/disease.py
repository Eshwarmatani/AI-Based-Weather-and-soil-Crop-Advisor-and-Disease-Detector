"""app/routes/disease.py — Plant disease detection API and page."""
import os
import uuid
from werkzeug.utils import secure_filename
from flask import Blueprint, render_template, request, jsonify
from app.app import db
from app.models import Prediction
import config

disease_bp = Blueprint("disease", __name__)

ALLOWED = config.ALLOWED_EXTENSIONS


def _allowed_file(filename: str | None) -> bool:
    if not filename or "." not in filename:
        return False
    return filename.rsplit(".", 1)[1].lower() in ALLOWED


@disease_bp.route("/disease-detection", methods=["GET"])
@disease_bp.route("/disease", methods=["GET"])
def disease_detection():
    return render_template("disease_detection.html")


@disease_bp.route("/api/disease/predict", methods=["POST"])
def api_disease_predict():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file provided"}), 400

    file = request.files.get("image")
    if not file or not file.filename:
        return jsonify({"success": False, "error": "No file selected"}), 400

    filename: str = file.filename
    if not _allowed_file(filename):
        return jsonify({
            "success": False,
            "error": f"Unsupported file type. Allowed: {ALLOWED}"
        }), 400

    # Save temporarily with secure random name
    safe_name = f"{uuid.uuid4().hex}_{secure_filename(filename)}"
    save_path = os.path.join(config.UPLOADS_DIR, safe_name)
    os.makedirs(config.UPLOADS_DIR, exist_ok=True)
    file.save(save_path)

    model_type = request.form.get("model_type", "garden").lower().strip()
    if model_type not in ["garden", "rice"]:
        model_type = "garden"

    try:
        if model_type == "rice":
            from src.disease.predict_rice import predict_rice_from_path
            result = predict_rice_from_path(save_path)
        else:
            from src.disease.predict import predict_from_path
            result = predict_from_path(save_path)
        result["model_type"] = model_type
    except FileNotFoundError as e:
        return jsonify({"success": False, "error": str(e)}), 503
    except Exception as e:
        return jsonify({"success": False, "error": f"Prediction failed: {str(e)}"}), 500
    finally:
        # Remove temp file safely after prediction
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
            except Exception:
                pass

    from ..services.advice_service import get_disease_advice
    advice = get_disease_advice(result["predicted_class"])

    # Store in history (no image stored)
    pred_id = None
    try:
        pred = Prediction(
            prediction_type=f"disease_{model_type}" if model_type == "rice" else "disease",
            disease_prediction=result["predicted_class"],
            confidence=result["confidence"],
        )
        db.session.add(pred)
        db.session.commit()
        pred_id = pred.id
    except Exception:
        db.session.rollback()

    return jsonify({"success": True, "result": result, "id": pred_id, "advice": advice})

