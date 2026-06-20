# 🚀 CardioGuard: Cardiometabolic Risk Stratification Engine

**Python • Scikit-Learn • XGBoost • FastAPI • Streamlit • SHAP**

**Predicting Cardiometabolic Disease Risk Using NHANES Data**

---

# 📋 Table of Contents

- Overview
- Business Context
- Key Features
- System Architecture
- Project Structure
- Technology Stack
- Quick Start
- Data Pipeline
- Model Development
- API Documentation
- Evaluation Metrics
- Clinical Impact
- Roadmap

---

# 🎯 Overview

CardioGuard is an end-to-end healthcare machine learning system designed to identify individuals at elevated cardiometabolic risk using demographic, anthropometric, laboratory, lifestyle, and blood pressure data.

This project demonstrates production-grade healthcare data science practices including:

✅ Cardiometabolic Risk Prediction

✅ Risk Stratification (Low / Moderate / High Risk)

✅ Advanced Feature Engineering

✅ Explainable AI using SHAP

✅ FastAPI REST API

✅ Streamlit Risk Assessment Dashboard

✅ End-to-End Deployment Pipeline

---

# 📊 Project Scope

| Metric | Value |
|----------|----------|
| Industry | Healthcare Analytics |
| Dataset | NHANES 2017–2018 |
| Population | US Adults |
| Records | ~1,850 participants |
| Features | 30+ engineered variables |
| Target | High Cardiometabolic Risk |
| Models | Logistic Regression, Random Forest, XGBoost |
| Deployment | FastAPI + Streamlit |

---

# 🏥 Business Context

## The Challenge

Cardiometabolic diseases remain among the leading causes of morbidity and mortality worldwide.

Healthcare providers often face:

📈 Increasing burden of diabetes

📈 Rising obesity prevalence

📈 Undiagnosed hypertension

📈 Delayed risk identification

📈 Limited prevention-focused analytics

---

## The Solution

CardioGuard provides:

- Early identification of high-risk individuals
- Automated risk stratification
- Real-time risk scoring
- Explainable predictions
- Population health insights

---

# ✨ Key Features

## 🎯 Cardiometabolic Risk Prediction

Predicts likelihood of high cardiometabolic risk.

Models evaluated:

- Logistic Regression
- Random Forest
- XGBoost

---

## 📊 Risk Stratification

Patients are grouped into:

| Risk Group | Description |
|------------|------------|
| Low Risk | Minimal intervention |
| Moderate Risk | Lifestyle modification |
| High Risk | Early clinical intervention |

---

## 🧠 Explainable AI

SHAP values used to explain:

- BMI impact
- Waist circumference
- Blood pressure contribution
- Glucose contribution
- Cholesterol contribution

---

## 🚀 Production API

Built with FastAPI.

Supports:

- Real-time scoring
- Risk classification
- JSON responses
- Interactive Swagger documentation

---

## 📈 Streamlit Dashboard

Interactive risk calculator:

- Patient inputs
- Risk probability
- Risk band classification
- Clinical interpretation

---

# 🏗️ System Architecture

```text
Raw NHANES Data
        │
        ▼
Preprocessing
        │
        ▼
Feature Engineering
        │
        ▼
Model Training
        │
        ▼
Risk Stratification
        │
        ▼
SHAP Explainability
        │
        ▼
FastAPI Deployment
        │
        ▼
Streamlit Dashboard
```

---

# 📁 Project Structure

```text
CardioGuard_Project/

│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_segmentation.ipynb
│   ├── 05_modeling.ipynb
│   ├── 06_explainability.ipynb
│   └── 07_dashboard_export.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── predict.py
│   │
│   └── api/
│       └── main.py
│
├── models/
│   ├── cardioguard_best_model.pkl
│   ├── cardioguard_feature_cols.pkl
│   ├── cardioguard_best_threshold.pkl
│   ├── cardioguard_kmeans_segmentation.pkl
│   ├── cardioguard_scaler_segmentation.pkl
│   └── cardioguard_segmentation_feature_cols.pkl
│
├── streamlit_app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 🛠 Technology Stack

| Category | Technologies |
|-----------|-------------|
| Machine Learning | Scikit-Learn, XGBoost |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Explainability | SHAP |
| API | FastAPI |
| Dashboard | Streamlit |
| Deployment | Uvicorn |

---

# 🚀 Quick Start

### Clone Repository

```bash
git clone https://github.com/Alpha-rammy/CardioGuard_Risk_Prediction.git
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📊 Data Pipeline

NHANES datasets used:

- Demographics
- Body Measures
- Blood Pressure
- Glucose
- HbA1c
- Total Cholesterol
- HDL Cholesterol
- Smoking
- Physical Activity

---

# 🔧 Feature Engineering

Features include:

### Anthropometric

- BMI
- Waist Circumference
- Waist-to-Height Ratio

### Blood Pressure

- Systolic BP
- Diastolic BP
- Pulse Pressure
- Mean Arterial Pressure

### Metabolic

- Fasting Glucose
- HbA1c
- Total Cholesterol
- HDL Cholesterol
- LDL Cholesterol

### Lifestyle

- Smoking Status
- Physical Activity

### Clinical Risk Indicators

- Obesity Flag
- Hypertension Flag
- Diabetes Flag
- Hyperlipidemia Flag

---

# 🤖 Model Development

## Models Evaluated

| Model | ROC-AUC |
|---------|---------|
| Logistic Regression | 0.8827 |
| Random Forest | 0.8895 |
| XGBoost | 0.9036 |

### Best Model

🏆 XGBoost

ROC-AUC = 0.9036

---

# 📡 API Documentation

### Start API

```bash
uvicorn src.api.main:app --reload
```

### Swagger Documentation

```text
http://127.0.0.1:8000/docs
```

### Example Response

```json
{
  "risk_probability": 0.87,
  "high_risk_prediction": 1,
  "risk_band": "High Risk"
}
```

---

# 📈 Evaluation Metrics

### XGBoost Performance

| Metric | Value |
|----------|----------|
| ROC-AUC | 0.9036 |
| Precision | 0.90 |
| Recall | 0.89 |
| F1 Score | 0.89 |

---

# 🩺 Clinical Impact

CardioGuard enables:

✅ Early disease prevention

✅ Population health monitoring

✅ Risk-based screening

✅ Clinical decision support

✅ Resource prioritization

---

# 🔮 Roadmap

### Phase 1

- Data Cleaning
- Feature Engineering
- Baseline Models

### Phase 2

- SHAP Explainability
- FastAPI Deployment
- Streamlit Dashboard

### Phase 3

- Docker Deployment
- Cloud Hosting
- NHS Integration Simulation

---

# 👨‍⚕️ Author

**Ransom Chukwu**

GitHub: https://github.com/Alpha-rammy

---