"""
app/routes/chat.py — AI Agriculture Chatbot API routes.
"""

import os
import uuid
from werkzeug.utils import secure_filename
from flask import Blueprint, request, jsonify

import config
from app.services.chat_service import process_chat_message

chat_bp = Blueprint("chat", __name__)


@chat_bp.route("/api/chat", methods=["POST"])
def api_chat():
    """
    Chatbot endpoint.
    Supports either JSON payload or multipart/form-data (with optional image file).
    """
    user_message: str = ""
    history: list = []
    model_type: str = "garden"
    uploaded_image_path: str | None = None

    if request.is_json:
        data = request.get_json(silent=True) or {}
        user_message = data.get("message", "")
        history = data.get("history", [])
        model_type = data.get("model_type", "garden")
    else:
        # Multipart form-data or urlencoded
        user_message = request.form.get("message", "")
        raw_hist = request.form.get("history", "")
        model_type = request.form.get("model_type", "garden")
        if raw_hist:
            import json
            try:
                history = json.loads(raw_hist)
            except Exception:
                history = []

        if "image" in request.files:
            file = request.files.get("image")
            if file and file.filename:
                filename: str = file.filename
                ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
                if ext not in config.ALLOWED_EXTENSIONS:
                    return jsonify({
                        "success": False,
                        "error": f"Invalid image format. Allowed: {config.ALLOWED_EXTENSIONS}"
                    }), 400

                os.makedirs(config.UPLOADS_DIR, exist_ok=True)
                safe_name = f"chat_{uuid.uuid4().hex}_{secure_filename(filename)}"
                uploaded_image_path = os.path.join(config.UPLOADS_DIR, safe_name)
                file.save(uploaded_image_path)

    # Validate message presence
    if not user_message and not uploaded_image_path:
        return jsonify({
            "success": False,
            "error": "Please provide a question or upload a leaf photo."
        }), 400

    try:
        result = process_chat_message(
            user_message=user_message,
            history=history,
            uploaded_image_path=uploaded_image_path,
            model_type=model_type
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Chatbot processing error: {str(e)}"
        }), 500
    finally:
        # Clean up temporary uploaded image file if any
        if uploaded_image_path and os.path.exists(uploaded_image_path):
            try:
                os.remove(uploaded_image_path)
            except Exception:
                pass


@chat_bp.route("/api/chat/status", methods=["GET"])
def api_chat_status():
    """Return active chatbot provider and configuration state."""
    active_provider = "local_expert_engine"
    if config.GEMINI_API_KEY:
        active_provider = f"Google Gemini ({config.CHATBOT_MODEL or 'gemini-1.5-flash'})"
    elif config.OPENAI_API_KEY:
        active_provider = f"OpenAI ({config.OPENAI_MODEL or 'gpt-4o-mini'})"

    return jsonify({
        "success": True,
        "configured_provider": config.CHATBOT_PROVIDER,
        "active_provider": active_provider,
        "has_gemini_key": bool(config.GEMINI_API_KEY),
        "has_openai_key": bool(config.OPENAI_API_KEY),
        "offline_knowledge_ready": True
    })
