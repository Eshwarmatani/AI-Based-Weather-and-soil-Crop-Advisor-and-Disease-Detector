import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.crop.predict import predict_top3
from src.weather_disease.predict import predict as predict_weather
from src.explainability.shap_explainer import explain_crop_dict, explain_weather_dict


class TestModels(unittest.TestCase):
    def test_crop_prediction(self):
        results = predict_top3(N=90, P=42, K=43, temperature=20.8, humidity=82.0, ph=6.5, rainfall=202.9)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["crop"], "rice")
        self.assertGreaterEqual(results[0]["probability"], 0.0)
        self.assertLessEqual(results[0]["probability"], 1.0)

    def test_weather_prediction(self):
        res = predict_weather(temperature=28.5, humidity=85.0, rainfall=15.0, wind_speed=12.0)
        self.assertIn("predicted_disease", res)
        self.assertGreaterEqual(res["probability"], 0.0)
        self.assertLessEqual(res["probability"], 1.0)
        self.assertIn("all_probs", res)

    def test_explainability_crop(self):
        shap_data = explain_crop_dict({
            "N": 90, "P": 42, "K": 43,
            "temperature": 20.8, "humidity": 82.0, "ph": 6.5, "rainfall": 202.9
        })
        self.assertIsNone(shap_data.get("error"))
        self.assertIn("shap_values", shap_data)
        self.assertEqual(len(shap_data["shap_values"]), 7)

    def test_explainability_weather(self):
        shap_data = explain_weather_dict({
            "temperature": 28.5, "humidity": 85.0, "rainfall": 15.0, "wind_speed": 12.0
        })
        self.assertIsNone(shap_data.get("error"))
        self.assertIn("shap_values", shap_data)
        self.assertEqual(len(shap_data["shap_values"]), 4)

    def test_plantvillage_disease_prediction(self):
        from src.disease.predict import predict_from_path
        sample_path = os.path.join(os.path.dirname(__file__), "..", "app", "static", "images", "samples", "tomato_late_blight.jpg")
        if os.path.exists(sample_path):
            res = predict_from_path(sample_path)
            self.assertIn("predicted_class", res)
            self.assertGreaterEqual(res["confidence"], 0.0)
            self.assertEqual(len(res["top3"]), 3)

    def test_rice_disease_prediction(self):
        from src.disease.predict_rice import predict_rice_from_path
        sample_path = os.path.join(os.path.dirname(__file__), "..", "app", "static", "images", "samples", "rice_bacterial_blight.jpg")
        if os.path.exists(sample_path):
            res = predict_rice_from_path(sample_path)
            self.assertIn("predicted_class", res)
            self.assertGreaterEqual(res["confidence"], 0.0)
            self.assertEqual(len(res["top3"]), 3)


if __name__ == "__main__":
    unittest.main()
