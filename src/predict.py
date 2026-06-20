import os
import joblib
import pandas as pd


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "cardioguard_best_model.pkl")
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "cardioguard_feature_cols.pkl")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "cardioguard_best_threshold.pkl")


model = joblib.load(MODEL_PATH)
feature_cols = joblib.load(FEATURE_COLS_PATH)
threshold = joblib.load(THRESHOLD_PATH)


def predict_risk(input_data: dict):
    """
    Predict cardiometabolic high-risk probability for one participant.
    """

    df = pd.DataFrame([input_data])

    # Ensure columns are in the same order used during training
    df = df[feature_cols]

    probability = model.predict_proba(df)[:, 1][0]

    prediction = int(probability >= threshold)

    if probability < 0.40:
        risk_band = "Low Risk"
    elif probability < 0.70:
        risk_band = "Moderate Risk"
    else:
        risk_band = "High Risk"

    return {
        "risk_probability": round(float(probability), 4),
        "high_risk_prediction": prediction,
        "risk_band": risk_band
    }


if __name__ == "__main__":

    sample_input = {
        "age": 55,
        "gender": "Male",
        "ethnicity": "Non-Hispanic Black",
        "education_level": "Some college",
        "poverty_income_ratio": 1.8,
        "age_65_plus_flag": 0,
        "low_income_flag": 0,
        "low_education_flag": 0,
        "bmi": 32.5,
        "waist_circumference": 105,
        "weight": 92,
        "height": 175,
        "waist_to_height_ratio": 105 / 175,
        "bmi_to_waist_ratio": 32.5 / 105,
        "systolic_bp": 145,
        "diastolic_bp": 92,
        "pulse_pressure": 145 - 92,
        "mean_arterial_pressure": ((2 * 92) + 145) / 3,
        "fasting_glucose": 130,
        "hba1c": 6.8,
        "glucose_hba1c_ratio": 130 / 6.8,
        "total_cholesterol": 220,
        "hdl_cholesterol": 38,
        "ldl_cholesterol": 145,
        "triglycerides": 180,
        "tc_hdl_ratio": 220 / 38,
        "ldl_hdl_ratio": 145 / 38,
        "trig_hdl_ratio": 180 / 38,
        "smoker_flag": 1,
        "physically_active_flag": 0,
        "inactive_flag": 1
    }

    result = predict_risk(sample_input)

    print(result)