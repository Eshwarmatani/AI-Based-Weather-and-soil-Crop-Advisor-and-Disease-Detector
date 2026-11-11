"""app/routes/weather.py — Weather-based disease risk API and page."""
import json
from flask import Blueprint, render_template, request, jsonify, redirect
from app.app import db
from app.models import Prediction

weather_bp = Blueprint("weather", __name__)


def _validate_weather_input(data: dict):
    required = ["temperature", "humidity", "rainfall", "wind_speed"]
    values = {}
    for field in required:
        val = data.get(field)
        if val is None or str(val).strip() == "":
            return None, f"Missing required field: {field}"
        try:
            values[field] = float(val)
        except ValueError:
            return None, f"Invalid value for {field}: must be a number"

    if not (0 <= values["temperature"] <= 50):
        return None, "Temperature must be between 0°C and 50°C"
    if not (0 <= values["humidity"] <= 100):
        return None, "Humidity must be between 0% and 100%"
    if not (0 <= values["rainfall"] <= 200):
        return None, "Rainfall must be between 0 and 200 mm/day"
    if not (0 <= values["wind_speed"] <= 100):
        return None, "Wind speed must be between 0 and 100 km/h"

    return values, None


@weather_bp.route("/weather-risk", methods=["GET"])
def weather_risk():
    return render_template("weather_risk.html")


@weather_bp.route("/api/weather/predict", methods=["POST"])
def api_weather_predict():
    data = request.get_json(silent=True) or request.form.to_dict()
    values, error = _validate_weather_input(data)
    if error:
        return jsonify({"success": False, "error": error}), 400

    try:
        from src.weather_disease.predict import predict
        result = predict(**values)
    except FileNotFoundError as e:
        return jsonify({"success": False, "error": str(e)}), 503

    # SHAP Explainability
    from src.explainability.shap_explainer import explain_weather_dict
    shap_data = explain_weather_dict(values)

    # Preventive Advice
    from ..services.advice_service import get_weather_risk_advice
    advice = get_weather_risk_advice(result["predicted_disease"])

    # Store in history
    pred_id = None
    try:
        pred = Prediction(
            prediction_type="weather",
            input_data=json.dumps(values),
            risk_prediction=result["predicted_disease"],
            confidence=result["probability"],
        )
        db.session.add(pred)
        db.session.commit()
        pred_id = pred.id
    except Exception:
        db.session.rollback()

    return jsonify({"success": True, "result": result, "id": pred_id, "explanation": shap_data, "advice": advice})


@weather_bp.route("/weather", methods=["GET"])
def weather_get():
    return redirect("/live-weather")


@weather_bp.route("/weather", methods=["POST"])
@weather_bp.route("/api/weather/live", methods=["POST"])
def weather():
    """Live weather API using Open-Meteo forecast service."""
    import requests

    data = request.get_json(silent=True) or {}
    latitude = data.get("latitude")
    longitude = data.get("longitude")

    if latitude is None or longitude is None:
        return jsonify({"error": "Location coordinates are missing"}), 400

    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": float(latitude),
            "longitude": float(longitude),
            "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code",
            "timezone": "auto"
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        weather_data = response.json()
        return jsonify(weather_data)
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to fetch live weather: {str(e)}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@weather_bp.route("/live-weather", methods=["GET"])
def live_weather_page():
    return render_template("live_weather.html")

