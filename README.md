# AI-Based Weather and Soil Crop Advisor & Disease Detector

> **An intelligent agriculture decision-support system featuring machine learning for crop recommendation, deep learning for plant leaf disease detection, and weather-driven disease risk assessment.**

[![Python](https://img.shields.io/badge/Python-3.10%2B%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask%203.1-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/DL-TensorFlow%20%2F%20Keras-red.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## 📌 Project Overview

This project provides an integrated, AI-driven advisory platform tailored for farmers, agronomists, and agricultural researchers. It addresses three critical challenges:

1. **Crop Recommendation**: Recommends the top-3 most suitable crops based on soil nutrients (N, P, K), soil pH, and local weather patterns (temperature, humidity, rainfall) with calibrated probability confidence.
2. **Plant Disease Identification**: Uses deep convolutional neural networks (MobileNetV2 transfer learning) to classify plant leaf diseases from user-uploaded images.
3. **Weather-Based Disease Risk Assessment**: Estimates the susceptibility of crops to specific diseases under prevailing environmental conditions (temperature, humidity, rainfall, wind speed).
4. **Interactive Web Dashboard**: An integrated Flask web portal featuring What-If simulations, model performance analytics, prediction logs, and SHAP explainability insights.

---

## 🏗 System Architecture

```text
                                  User Inputs
          [ Soil / Weather Parameters ]         [ Leaf Image Upload ]
                       |                                  |
                       v                                  v
            +---------------------+            +-----------------------+
            |      MODEL 1:       |            |       MODEL 2:        |
            | Crop Recommendation |            | Plant Disease CNN     |
            |   (scikit-learn)    |            |   (MobileNetV2)       |
            +---------------------+            +-----------------------+
                       |                                  |
                       |       +--------------------+     |
                       +-----> |     MODEL 3:       | <---+
                               | Weather-Based Risk |
                               |   (scikit-learn)   |
                               +--------------------+
                                         |
                                         v
                         +-------------------------------+
                         |      FLASK WEB DASHBOARD      |
                         |  - Predictions & Probabilities|
                         |  - Model Performance & Plots  |
                         |  - What-If Simulator & History|
                         +-------------------------------+
```

---

## 🤖 Models & Methodologies

| Component | Architecture / Model | Input Features | Performance / Metric |
|---|---|---|---|
| **Crop Advisor** | Tuned scikit-learn Pipeline (StandardScaler + Best Classifier) | N, P, K, Temp, Humidity, pH, Rainfall | Macro F1 & Accuracy (~99%) |
| **Plant Disease** | MobileNetV2 (Transfer Learning with Fine-Tuning) | 224x224 RGB Leaf Images | High Accuracy on PlantVillage & Rice Disease |
| **Weather Risk** | Environmental Risk Classifier | Temp, Humidity, Rainfall, Wind Speed | Calibrated Risk Probabilities |

---

## 📂 Repository Structure

```text
.
├── AI_Crop_Advisor/
│   ├── app/                 # Flask application factory, blueprints, routes, templates, static
│   ├── config.py            # Central path & configuration registry
│   ├── data/                # Tabular datasets & processed data splits (CSVs & NPYs)
│   ├── database/            # SQLite prediction history database
│   ├── models/              # Pre-trained model artifacts (.joblib, .keras, class maps)
│   ├── notebooks/           # Jupyter exploratory notebooks
│   ├── reports/             # Generated evaluation reports, metrics & plots
│   ├── requirements.txt     # Python dependency specifications
│   ├── run.py               # Flask development web server entry point
│   ├── src/                 # ML pipelines: crop, disease, weather_disease modules
│   ├── tests/               # Test suites (unit and integration tests)
│   └── train_all.py         # Complete sequential pipeline training runner
├── download_datasets.py     # Helper script to download Kaggle datasets
├── smoke_test.py            # Automated route and API smoke verification
├── test_all_models.py       # End-to-end inference test for all ML models
├── scratch_eval.py          # Standalone hold-out evaluation script
├── pyproject.toml           # Project configuration
├── requirements.txt         # Root requirements file
└── README.md                # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Eshwarmatani/AI-Based-Weather-and-soil-Crop-Advisor-and-Disease-Detector.git
cd AI-Based-Weather-and-soil-Crop-Advisor-and-Disease-Detector
```

### 2. Set Up Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r AI_Crop_Advisor/requirements.txt
```

### 4. Run the Web Application
Pre-trained model artifacts are pre-bundled in `AI_Crop_Advisor/models/`, allowing immediate execution:
```bash
python AI_Crop_Advisor/run.py
```
Open your browser and navigate to **`http://localhost:5000`**.

---

## 🧪 Verification & Testing

Verify that all models and web endpoints function properly:

- **End-to-End Model Test**:
  ```bash
  python test_all_models.py
  ```
- **Flask Route & API Smoke Test**:
  *(Ensure Flask app is running)*
  ```bash
  python smoke_test.py
  ```
- **Hold-out Evaluation**:
  ```bash
  python scratch_eval.py
  ```

---

## 📈 Training from Scratch (Optional)

To re-train all models on the latest dataset splits:
```bash
cd AI_Crop_Advisor
python train_all.py
```
Or train individual components:
- Crop Model: `python src/crop/train.py`
- Disease CNN: `python src/disease/train.py`
- Weather Disease Model: `python src/weather_disease/train.py`

---

## 📜 Disclaimer

This project is an academic decision-support tool. Crop recommendations and disease detections are provided as statistical guidance based on trained historical distributions and should be cross-referenced with local agricultural extension services before large-scale agronomic interventions.
