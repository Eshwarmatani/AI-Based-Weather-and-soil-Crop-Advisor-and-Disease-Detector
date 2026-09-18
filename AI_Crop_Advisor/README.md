# AI-Based Weather and Soil Crop Advisor
## with Plant Disease Detection and Weather-Based Disease Risk Prediction

> University Final/FIP Machine Learning Project

---

## Project Objective

Build an intelligent agriculture decision-support system that answers four questions:

1. Which crop is most suitable for the given soil and weather conditions?
2. What disease is present in an uploaded plant-leaf image?
3. Based on selected crop and weather conditions, what is the risk of selected crop diseases?
4. What general management advice should the user consider?

---

## Features

| Feature | Description |
|---|---|
| Crop Recommendation | Top-3 crops with confidence scores from soil/weather input |
| Plant Disease Detection | CNN-based image classification using MobileNetV2 |
| Weather Disease Risk | ML-based risk estimation from weather features |
| Dashboard | Integrated advisory dashboard |
| Explainability | SHAP feature importance for tabular models |
| What-If Simulator | Real-time model re-prediction on changed inputs |
| History | SQLite database of all predictions |
| Model Performance | Compare all models with real metrics |

---

## Architecture

```
User Input (Soil/Weather/Image)
        |
        |-- MODEL 1: Crop Recommendation (scikit-learn)
        |       |-- Top-3 crops + confidence
        |
        |-- MODEL 2: Plant Disease Detection (MobileNetV2 / TensorFlow)
        |       |-- Disease class + confidence
        |
        |-- MODEL 3: Weather-Based Disease Risk (scikit-learn)
                |-- Risk class from trained historical model
                        |
                        |-- INTEGRATED DASHBOARD
```

---

## Datasets

| Dataset | Source | Records | Task |
|---|---|---|---|
| Crop Recommendation | Kaggle: atharvaingle/crop-recommendation-dataset | 2,200 rows | Crop classification |
| Plant Disease (PlantVillage) | Kaggle: emmarex/plantdisease | ~41,278 images, 15 classes | Image classification |
| Rice Disease (Image-based) | Kaggle: anshulm257/rice-disease-dataset | ~3,829 images, 6 classes | Image + risk classification |

NOTE: The Rice Disease dataset contains image classes, not tabular weather-disease records.
Model 3 (Weather-Based Disease Risk) will be trained on features derived from rice disease image
classes using scientifically-grounded weather feature ranges per disease class.
See Phase 2 report for final decision and methodology.

---

## ML Algorithms

### Model 1 - Crop Recommendation
- Logistic Regression
- Decision Tree
- Random Forest (comparison only - winner determined by test data)
- K-Nearest Neighbors
- Support Vector Machine
- Gradient Boosting

### Model 2 - Plant Disease Detection
- MobileNetV2 (Transfer Learning) - Primary
- EfficientNetB0 - Optional comparison

### Model 3 - Weather-Based Disease Risk
- Logistic Regression
- Decision Tree
- Random Forest
- SVM
- Gradient Boosting

---

## Installation

### Prerequisites
- Python 3.13.x
- Windows / Linux / macOS
- 8 GB RAM minimum (16 GB recommended for image training)

### Setup

```
cd AI_Crop_Advisor
python -m venv venv
venv\Scripts\activate     (Windows)
pip install -r requirements.txt
```

---

## Training Instructions

Run phases in order. Do not skip phases.

```
python src/crop/train.py
python src/disease/train.py
python src/weather_disease/train.py
```

---

## Running the Application

```
python run.py
```
Open browser at: http://localhost:5000

---

## Dataset Paths (Local Kaggle Cache)

| Dataset | Local Path |
|---|---|
| Crop Recommendation | %USERPROFILE%\.cache\kagglehub\datasets\atharvaingle\crop-recommendation-dataset\versions\1\Crop_recommendation.csv |
| Plant Disease | %USERPROFILE%\.cache\kagglehub\datasets\emmarex\plantdisease\versions\1\PlantVillage\ |
| Rice Disease Images | %USERPROFILE%\.cache\kagglehub\datasets\anshulm257\rice-disease-dataset\versions\1\Rice_Leaf_AUG\ |

---

## Important Limitations

1. PlantVillage accuracy does not equal real farm accuracy. Images are lab-quality.
2. Weather disease model is based on rice image class distributions, not real agro-meteorological records.
3. The system does NOT prescribe pesticide dosages. All guidance is general management advice.
4. Weather-risk prediction currently supports only rice disease classes.
5. No external API is required. All predictions are from locally trained models.

---

## Project Status

| Phase | Status |
|---|---|
| Phase 1 - Project Inspection | Complete |
| Phase 2 - Dataset Setup | Awaiting approval |
| Phase 3 - Crop Model | Pending |
| Phase 4 - Disease Image Model | Pending |
| Phase 5 - Weather Disease Model | Pending |
| Phase 6 - Model Integration | Pending |
| Phase 7 - Flask Application | Pending |
| Phase 8 - Advanced Features | Pending |
| Phase 9 - Testing | Pending |
| Phase 10 - Final Documentation | Pending |
