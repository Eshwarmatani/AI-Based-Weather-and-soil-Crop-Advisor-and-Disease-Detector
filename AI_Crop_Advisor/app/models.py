"""
app/models.py — SQLAlchemy database models.
"""

import json
from datetime import datetime, timezone
from app.app import db


class Prediction(db.Model):
    __tablename__ = "predictions"

    id              = db.Column(db.Integer, primary_key=True)
    timestamp       = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    prediction_type = db.Column(db.String(50), nullable=False)   # 'crop', 'disease', 'weather', 'combined'
    input_data      = db.Column(db.Text, nullable=True)           # JSON string
    crop_prediction  = db.Column(db.String(200), nullable=True)
    disease_prediction = db.Column(db.String(200), nullable=True)
    risk_prediction  = db.Column(db.String(100), nullable=True)
    confidence       = db.Column(db.Float, nullable=True)
    created_at       = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def __init__(
        self,
        prediction_type: str = "",
        input_data: str | None = None,
        crop_prediction: str | None = None,
        disease_prediction: str | None = None,
        risk_prediction: str | None = None,
        confidence: float | None = None,
        **kwargs,
    ):
        super().__init__()
        self.prediction_type = prediction_type
        self.input_data = input_data
        self.crop_prediction = crop_prediction
        self.disease_prediction = disease_prediction
        self.risk_prediction = risk_prediction
        self.confidence = confidence
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self):
        parsed_input = {}
        if self.input_data:
            try:
                parsed_input = json.loads(self.input_data)
            except Exception:
                parsed_input = {"raw": self.input_data}
        return {
            "id":                 self.id,
            "timestamp":          self.timestamp.isoformat() if self.timestamp else None,
            "prediction_type":    self.prediction_type,
            "input_data":         parsed_input,
            "crop_prediction":    self.crop_prediction,
            "disease_prediction": self.disease_prediction,
            "risk_prediction":    self.risk_prediction,
            "confidence":         self.confidence,
        }
