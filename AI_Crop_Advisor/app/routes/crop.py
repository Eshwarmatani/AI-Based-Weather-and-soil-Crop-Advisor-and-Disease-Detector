"""app/routes/crop.py — Crop recommendation API and page."""
import json
from flask import Blueprint, render_template, request, jsonify
from app.app import db
from app.models import Prediction
from src.crop.predict import predict_top3

crop_bp = Blueprint("crop", __name__)


def _validate_crop_input(data: dict) -> tuple[dict | None, str | None]:
    """Validate and parse crop form inputs. Returns (values_dict, error_message)."""
    required = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    values = {}
    for field in required:
        val = data.get(field)
        if val is None or str(val).strip() == "":
            return None, f"Missing required field: {field}"
        try:
            values[field] = float(val)
        except ValueError:
            return None, f"Invalid value for {field}: must be a number"

    # Domain validation
    if not (0 <= values["N"] <= 140):
        return None, "Nitrogen (N) must be between 0 and 140"
    if not (0 <= values["P"] <= 145):
        return None, "Phosphorus (P) must be between 0 and 145"
    if not (0 <= values["K"] <= 205):
        return None, "Potassium (K) must be between 0 and 205"
    if not (0 <= values["temperature"] <= 50):
        return None, "Temperature must be between 0°C and 50°C"
    if not (0 <= values["humidity"] <= 100):
        return None, "Humidity must be between 0% and 100%"
    if not (3 <= values["ph"] <= 10):
        return None, "Soil pH must be between 3 and 10"
    if not (0 <= values["rainfall"] <= 300):
        return None, "Rainfall must be between 0 and 300 mm"

    return values, None


@crop_bp.route("/crop-advisor", methods=["GET"])
@crop_bp.route("/crop-recommendation", methods=["GET"])
@crop_bp.route("/crop", methods=["GET"])
def crop_advisor():
    return render_template("crop_advisor.html")


@crop_bp.route("/api/crop/predict", methods=["POST"])
def api_crop_predict():
    data = request.get_json(silent=True) or request.form.to_dict()
    values, error = _validate_crop_input(data)
    if error or values is None:
        return jsonify({"success": False, "error": error or "Invalid input"}), 400

    try:
        results = predict_top3(
            N=values["N"],
            P=values["P"],
            K=values["K"],
            temperature=values["temperature"],
            humidity=values["humidity"],
            ph=values["ph"],
            rainfall=values["rainfall"],
        )
    except FileNotFoundError as e:
        return jsonify({"success": False, "error": str(e)}), 503

    # SHAP Explainability
    from src.explainability.shap_explainer import explain_crop_dict
    shap_data = explain_crop_dict(values)

    # Agronomic Guidance
    from ..services.advice_service import get_crop_guidance
    crop_guidance = get_crop_guidance(results[0]["crop"], values)

    # Store in history
    pred_id = None
    try:
        pred = Prediction(
            prediction_type="crop",
            input_data=json.dumps(values),
            crop_prediction=results[0]["crop"],
            confidence=results[0]["probability"],
        )
        db.session.add(pred)
        db.session.commit()
        pred_id = pred.id
    except Exception:
        db.session.rollback()

    return jsonify({
        "success": True,
        "results": results,
        "result": results,
        "id": pred_id,
        "explanation": shap_data,
        "guidance": crop_guidance
    })
