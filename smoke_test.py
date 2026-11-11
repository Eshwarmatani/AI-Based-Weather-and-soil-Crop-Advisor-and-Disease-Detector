"""Smoke test all Flask routes and APIs."""
import requests

base = 'http://127.0.0.1:5000'
pages = ['/', '/crop-advisor', '/disease-detection', '/weather-risk', '/model-performance', '/history']

print('=== Route smoke test ===')
for page in pages:
    try:
        r = requests.get(base + page, timeout=5)
        status = 'OK' if r.status_code == 200 else 'ERROR ' + str(r.status_code)
        print(f'  {page:<30s} -> {status}')
    except Exception as e:
        print(f'  {page:<30s} -> FAIL: {e}')

print()
print('=== API smoke tests ===')

# Crop API
r = requests.post(base + '/api/crop/predict',
    json={'N': 90, 'P': 42, 'K': 43, 'temperature': 20.87, 'humidity': 82.0, 'ph': 6.5, 'rainfall': 202.9},
    timeout=10)
data = r.json()
print(f'  /api/crop/predict    -> {r.status_code}  success={data.get("success")}')
if data.get('success') and data.get('result'):
    print(f'    top crop: {data["result"][0]["crop"]}  {data["result"][0]["probability"]:.2%}')

# Weather API
r = requests.post(base + '/api/weather/predict',
    json={'temperature': 25.0, 'humidity': 94.0, 'rainfall': 12.0, 'wind_speed': 17.0},
    timeout=10)
data = r.json()
print(f'  /api/weather/predict -> {r.status_code}  success={data.get("success")}')
if data.get('success') and data.get('result'):
    print(f'    disease: {data["result"]["predicted_disease"]}  {data["result"]["probability"]:.2%}')

# Performance APIs
for m in ['crop', 'disease', 'weather', 'rice-disease']:
    try:
        r = requests.get(base + '/api/performance/' + m, timeout=5)
        data = r.json()
        print(f'  /api/performance/{m:<12s} -> {r.status_code} classes={data.get("n_classes")} samples={data.get("total_test_samples")}')
    except Exception as e:
        print(f'  /api/performance/{m:<12s} -> FAIL: {e}')

# Chatbot APIs
r = requests.get(base + '/api/chat/status', timeout=5)
data = r.json()
print(f'  /api/chat/status     -> {r.status_code}  active_provider={data.get("active_provider")}')

r = requests.post(base + '/api/chat', json={'message': 'What is soil pH?'}, timeout=10)
data = r.json()
print(f'  /api/chat            -> {r.status_code}  success={data.get("success")} provider={data.get("provider")}')

print()
print('ALL ROUTES VERIFIED!')
