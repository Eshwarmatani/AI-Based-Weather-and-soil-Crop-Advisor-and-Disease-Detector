"""app/routes/history.py — Prediction history page."""
from flask import Blueprint, render_template, jsonify
from app.app import db
from app.models import Prediction

history_bp = Blueprint("history", __name__)


@history_bp.route("/history")
def history():
    return render_template("history.html")


@history_bp.route("/api/history")
def api_history():
    try:
        preds = Prediction.query.order_by(Prediction.created_at.desc()).limit(100).all()
        return jsonify([p.to_dict() for p in preds])
    except Exception:
        return jsonify([]), 200


@history_bp.route("/api/history/clear", methods=["POST"])
def api_history_clear():
    try:
        num_deleted = Prediction.query.delete()
        db.session.commit()
        return jsonify({"success": True, "deleted": num_deleted})
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)}), 500


@history_bp.route("/api/history/export")
def api_history_export():
    import csv
    import io
    from flask import Response

    preds = Prediction.query.order_by(Prediction.created_at.desc()).all()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Timestamp", "Type", "Crop", "Disease", "Risk", "Confidence", "Input_Data"])

    for p in preds:
        writer.writerow([
            p.id,
            p.timestamp.isoformat() if p.timestamp else "",
            p.prediction_type,
            p.crop_prediction or "",
            p.disease_prediction or "",
            p.risk_prediction or "",
            f"{p.confidence:.4f}" if p.confidence is not None else "",
            p.input_data or ""
        ])

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=prediction_history.csv"}
    )
