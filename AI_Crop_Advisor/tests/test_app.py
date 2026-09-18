import unittest
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.app import create_app


class TestApp(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def test_routes_status(self):
        routes = [
            "/",
            "/crop-advisor",
            "/disease-detection",
            "/weather-risk",
            "/dashboard",
            "/history",
            "/model-performance",
            "/about",
        ]
        for r in routes:
            resp = self.client.get(r)
            self.assertEqual(resp.status_code, 200, f"Route {r} failed with {resp.status_code}")

    def test_api_performance_endpoints(self):
        for ep in ["/api/performance/crop", "/api/performance/weather", "/api/performance/disease", "/api/performance/rice-disease"]:
            resp = self.client.get(ep)
            self.assertEqual(resp.status_code, 200, f"Endpoint {ep} failed with {resp.status_code}")

        crop_resp = self.client.get("/api/performance/crop").get_json()
        self.assertIn("results", crop_resp)
        self.assertGreater(len(crop_resp["results"]), 0)
        for r in crop_resp["results"]:
            self.assertIn("model", r)
            self.assertIn("accuracy", r)
            self.assertIn("precision", r)
            self.assertIn("recall", r)
            self.assertIn("f1", r)
            self.assertIn("train_time_sec", r)

    def test_crop_api_predict_and_guidance(self):
        payload = {
            "N": 90, "P": 42, "K": 43,
            "temperature": 20.8, "humidity": 82.0, "ph": 6.5, "rainfall": 202.9
        }
        resp = self.client.post("/api/crop/predict", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(len(data["results"]), 3)
        self.assertIsNotNone(data.get("explanation"))
        self.assertIsNone(data["explanation"].get("error"))
        self.assertIn("guidance", data)
        self.assertIn("profile", data["guidance"])

    def test_crop_api_validation(self):
        resp = self.client.post("/api/crop/predict", json={"N": 90})
        self.assertEqual(resp.status_code, 400)
        self.assertFalse(resp.get_json()["success"])

        resp = self.client.post("/api/crop/predict", json={
            "N": 999, "P": 42, "K": 43,
            "temperature": 20.8, "humidity": 82.0, "ph": 6.5, "rainfall": 202.9
        })
        self.assertEqual(resp.status_code, 400)
        self.assertFalse(resp.get_json()["success"])

    def test_weather_api_predict_and_advice(self):
        payload = {
            "temperature": 28.5, "humidity": 85.0, "rainfall": 15.0, "wind_speed": 12.0
        }
        resp = self.client.post("/api/weather/predict", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertIn("predicted_disease", data["result"])
        self.assertIsNone(data.get("explanation", {}).get("error"))
        self.assertIn("advice", data)
        self.assertIn("preventive_actions", data["advice"])

    def test_history_endpoints(self):
        # Fetch list
        resp = self.client.get("/api/history")
        self.assertEqual(resp.status_code, 200)
        self.assertIsInstance(resp.get_json(), list)

        # Export CSV
        resp_export = self.client.get("/api/history/export")
        self.assertEqual(resp_export.status_code, 200)
        self.assertEqual(resp_export.mimetype, "text/csv")
        self.assertIn(b"Timestamp", resp_export.data)

        # Clear history
        resp_clear = self.client.post("/api/history/clear")
        self.assertEqual(resp_clear.status_code, 200)
        self.assertTrue(resp_clear.get_json()["success"])

    def test_live_weather_endpoints(self):
        # GET live weather page
        resp_page = self.client.get("/live-weather")
        self.assertEqual(resp_page.status_code, 200)

        # POST /weather missing coordinates
        resp_bad = self.client.post("/weather", json={})
        self.assertEqual(resp_bad.status_code, 400)
        self.assertIn("error", resp_bad.get_json())

        # POST /weather with coordinates
        resp_live = self.client.post("/weather", json={"latitude": 28.6139, "longitude": 77.2090})
        self.assertEqual(resp_live.status_code, 200)
        data = resp_live.get_json()
        self.assertIn("current", data)
        self.assertIn("temperature_2m", data["current"])
        self.assertIn("relative_humidity_2m", data["current"])

    def test_disease_api_predict_both_models(self):
        samples_dir = os.path.join(os.path.dirname(__file__), "..", "app", "static", "images", "samples")
        
        # Test Garden Model
        tomato_path = os.path.join(samples_dir, "tomato_late_blight.jpg")
        if os.path.exists(tomato_path):
            with open(tomato_path, "rb") as f:
                resp = self.client.post(
                    "/api/disease/predict",
                    data={"image": (f, "tomato.jpg"), "model_type": "garden"},
                    content_type="multipart/form-data"
                )
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertTrue(data["success"])
            self.assertEqual(data["result"]["model_type"], "garden")
            self.assertIn("advice", data)

        # Test Rice Model
        rice_path = os.path.join(samples_dir, "rice_bacterial_blight.jpg")
        if os.path.exists(rice_path):
            with open(rice_path, "rb") as f:
                resp = self.client.post(
                    "/api/disease/predict",
                    data={"image": (f, "rice.jpg"), "model_type": "rice"},
                    content_type="multipart/form-data"
                )
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertTrue(data["success"])
            self.assertEqual(data["result"]["model_type"], "rice")
            self.assertIn("advice", data)


if __name__ == "__main__":
    unittest.main()
