import requests

base_url = "http://127.0.0.1:5000"

# 1. HTML Pages
print("Checking /disease-detection...")
r = requests.get(f"{base_url}/disease-detection")
assert r.status_code == 200
assert "modelTabGarden" in r.text
assert "modelTabRice" in r.text
assert "samplesRice" in r.text
print("-> OK: disease-detection HTML contains model switcher and samples.")

# 2. Rice Disease Performance API
print("Checking /api/performance/rice-disease...")
r = requests.get(f"{base_url}/api/performance/rice-disease")
assert r.status_code == 200
data = r.json()
acc = data.get("test_accuracy", 0) * 100
print(f"-> OK: Rice test accuracy: {acc:.2f}%, classes: {data.get('n_classes')}")

# 3. Predict Rice
print("Testing POST /api/disease/predict with model_type=rice...")
with open("AI_Crop_Advisor/app/static/images/samples/rice_bacterial_blight.jpg", "rb") as f:
    r = requests.post(f"{base_url}/api/disease/predict", files={"image": f}, data={"model_type": "rice"})
assert r.status_code == 200
res = r.json()
pred_cls = res["result"]["predicted_class"]
conf = res["result"]["confidence"] * 100
print(f"-> OK: Rice Prediction: {pred_cls} ({conf:.1f}%)")
print(f"-> OK: Rice Advice Title: {res['advice']['title']}")

# 4. Predict Garden
print("Testing POST /api/disease/predict with model_type=garden...")
with open("AI_Crop_Advisor/app/static/images/samples/tomato_late_blight.jpg", "rb") as f:
    r = requests.post(f"{base_url}/api/disease/predict", files={"image": f}, data={"model_type": "garden"})
assert r.status_code == 200
res = r.json()
pred_cls = res["result"]["predicted_class"]
conf = res["result"]["confidence"] * 100
print(f"-> OK: Garden Prediction: {pred_cls} ({conf:.1f}%)")
print(f"-> OK: Garden Advice Title: {res['advice']['title']}")
print("\nALL LIVE ENDPOINTS VERIFIED AND WORKING PERFECTLY!")
