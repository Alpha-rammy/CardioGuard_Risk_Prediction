import os
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(DATA_PROCESSED, exist_ok=True)


def load_clean_data():
    return {
        "demo": pd.read_csv(os.path.join(DATA_PROCESSED, "demo_clean.csv")),
        "bmx": pd.read_csv(os.path.join(DATA_PROCESSED, "bmx_clean.csv")),
        "bpx": pd.read_csv(os.path.join(DATA_PROCESSED, "bpx_clean.csv")),
        "glu": pd.read_csv(os.path.join(DATA_PROCESSED, "glu_clean.csv")),
        "ghb": pd.read_csv(os.path.join(DATA_PROCESSED, "ghb_clean.csv")),
        "tchol": pd.read_csv(os.path.join(DATA_PROCESSED, "tchol_clean.csv")),
        "hdl": pd.read_csv(os.path.join(DATA_PROCESSED, "hdl_clean.csv")),
        "trigly": pd.read_csv(os.path.join(DATA_PROCESSED, "trigly_clean.csv")),
        "smq": pd.read_csv(os.path.join(DATA_PROCESSED, "smq_clean.csv")),
        "paq": pd.read_csv(os.path.join(DATA_PROCESSED, "paq_clean.csv")),
        "bpq": pd.read_csv(os.path.join(DATA_PROCESSED, "bpq_clean.csv")),
        "diq": pd.read_csv(os.path.join(DATA_PROCESSED, "diq_clean.csv")),
    }


def build_features():
    data = load_clean_data()

    demo = data["demo"]
    bmx = data["bmx"]
    bpx = data["bpx"]
    glu = data["glu"]
    ghb = data["ghb"]
    tchol = data["tchol"]
    hdl = data["hdl"]
    trigly = data["trigly"]
    smq = data["smq"]
    paq = data["paq"]
    bpq = data["bpq"]
    diq = data["diq"]

    base = demo[["participant_id"]].copy()

    # Anthropometric features
    anthro = bmx.copy()

    anthro["waist_to_height_ratio"] = (
        anthro["waist_circumference"] / anthro["height"]
    )

    anthro["bmi_to_waist_ratio"] = (
        anthro["bmi"] / anthro["waist_circumference"]
    )

    anthro = anthro[
        [
            "participant_id",
            "bmi",
            "waist_circumference",
            "weight",
            "height",
            "waist_to_height_ratio",
            "bmi_to_waist_ratio",
        ]
    ]

    # Cardiovascular features
    cardio = bpx.copy()

    cardio["pulse_pressure"] = (
        cardio["systolic_bp"] - cardio["diastolic_bp"]
    )

    cardio["mean_arterial_pressure"] = (
        (2 * cardio["diastolic_bp"] + cardio["systolic_bp"]) / 3
    )

    cardio = cardio[
        [
            "participant_id",
            "systolic_bp",
            "diastolic_bp",
            "pulse_pressure",
            "mean_arterial_pressure",
        ]
    ]

    # Glycemic features
    glycemic = glu.merge(
        ghb,
        on="participant_id",
        how="outer"
    )

    glycemic["glucose_hba1c_ratio"] = (
        glycemic["fasting_glucose"] / glycemic["hba1c"]
    )

    glycemic = glycemic[
        [
            "participant_id",
            "fasting_glucose",
            "hba1c",
            "glucose_hba1c_ratio",
        ]
    ]

    # Lipid features
    lipids = (
        tchol
        .merge(hdl, on="participant_id", how="outer")
        .merge(trigly, on="participant_id", how="outer")
    )

    lipids["tc_hdl_ratio"] = (
        lipids["total_cholesterol"] / lipids["hdl_cholesterol"]
    )

    lipids["ldl_hdl_ratio"] = (
        lipids["ldl_cholesterol"] / lipids["hdl_cholesterol"]
    )

    lipids["trig_hdl_ratio"] = (
        lipids["triglycerides"] / lipids["hdl_cholesterol"]
    )

    lipids = lipids[
        [
            "participant_id",
            "total_cholesterol",
            "hdl_cholesterol",
            "ldl_cholesterol",
            "triglycerides",
            "tc_hdl_ratio",
            "ldl_hdl_ratio",
            "trig_hdl_ratio",
        ]
    ]

    # Lifestyle features
    lifestyle = (
        smq[["participant_id", "smoking_status"]]
        .merge(
            paq[["participant_id", "vigorous_activity", "moderate_activity"]],
            on="participant_id",
            how="outer",
        )
    )

    lifestyle["smoker_flag"] = (
        lifestyle["smoking_status"] == "Yes"
    ).astype(int)

    lifestyle["physically_active_flag"] = (
        (lifestyle["vigorous_activity"] == "Yes") |
        (lifestyle["moderate_activity"] == "Yes")
    ).astype(int)

    lifestyle["inactive_flag"] = (
        lifestyle["physically_active_flag"] == 0
    ).astype(int)

    lifestyle = lifestyle[
        [
            "participant_id",
            "smoker_flag",
            "physically_active_flag",
            "inactive_flag",
        ]
    ]

    # Medical history features
    history = (
        bpq[["participant_id", "hypertension_history", "bp_medication"]]
        .merge(
            diq[["participant_id", "diabetes_history"]],
            on="participant_id",
            how="outer",
        )
    )

    history["hypertension_history_flag"] = (
        history["hypertension_history"] == "Yes"
    ).astype(int)

    history["bp_medication_flag"] = (
        history["bp_medication"] == "Yes"
    ).astype(int)

    history["diabetes_history_flag"] = (
        history["diabetes_history"] == "Yes"
    ).astype(int)

    history = history[
        [
            "participant_id",
            "hypertension_history_flag",
            "bp_medication_flag",
            "diabetes_history_flag",
        ]
    ]

    # Socioeconomic features
    socio = demo.copy()

    socio["age_65_plus_flag"] = (
        socio["age"] >= 65
    ).astype(int)

    socio["low_income_flag"] = (
        socio["poverty_income_ratio"] < 1
    ).astype(int)

    socio["low_education_flag"] = (
        socio["education_level"].isin([
            "Less than 9th grade",
            "9th-11th grade"
        ])
    ).astype(int)

    socio = socio[
        [
            "participant_id",
            "age",
            "gender",
            "ethnicity",
            "education_level",
            "poverty_income_ratio",
            "age_65_plus_flag",
            "low_income_flag",
            "low_education_flag",
        ]
    ]

    # Merge predictor groups
    features = (
        base
        .merge(socio, on="participant_id", how="left")
        .merge(anthro, on="participant_id", how="left")
        .merge(cardio, on="participant_id", how="left")
        .merge(glycemic, on="participant_id", how="left")
        .merge(lipids, on="participant_id", how="left")
        .merge(lifestyle, on="participant_id", how="left")
        .merge(history, on="participant_id", how="left")
    )

    # Create target separately
    target_flags = features[["participant_id"]].copy()

    target_flags["obesity_target_flag"] = (
        features["bmi"] >= 30
    ).astype(int)

    target_flags["hypertension_target_flag"] = (
        (features["systolic_bp"] >= 130) |
        (features["diastolic_bp"] >= 80) |
        (features["hypertension_history_flag"] == 1) |
        (features["bp_medication_flag"] == 1)
    ).astype(int)

    target_flags["diabetes_target_flag"] = (
        (features["fasting_glucose"] >= 126) |
        (features["hba1c"] >= 6.5) |
        (features["diabetes_history_flag"] == 1)
    ).astype(int)

    target_flags["dyslipidemia_target_flag"] = (
        (features["total_cholesterol"] >= 200) |
        (features["ldl_cholesterol"] >= 130) |
        (features["hdl_cholesterol"] < 40) |
        (features["triglycerides"] >= 150)
    ).astype(int)

    target_flags["target_risk_count"] = (
        target_flags["obesity_target_flag"] +
        target_flags["hypertension_target_flag"] +
        target_flags["diabetes_target_flag"] +
        target_flags["dyslipidemia_target_flag"]
    )

    target_flags["high_risk_flag"] = (
        target_flags["target_risk_count"] >= 3
    ).astype(int)

    features = features.merge(
        target_flags[["participant_id", "high_risk_flag"]],
        on="participant_id",
        how="left"
    )

    # Remove target-leakage columns
    leakage_cols = [
        "hypertension_history_flag",
        "bp_medication_flag",
        "diabetes_history_flag",
    ]

    features = features.drop(
        columns=[col for col in leakage_cols if col in features.columns]
    )

    # Handle infinite values from ratios
    features = features.replace([np.inf, -np.inf], np.nan)

    output_path = os.path.join(
        DATA_PROCESSED,
        "cardioguard_model_features.csv"
    )

    features.to_csv(output_path, index=False)

    print("Feature engineering completed.")
    print(f"Saved: {output_path}")
    print(f"Rows: {features.shape[0]:,}")
    print(f"Columns: {features.shape[1]:,}")
    print("\nTarget distribution:")
    print(features["high_risk_flag"].value_counts(normalize=True).mul(100).round(2))

    return features


if __name__ == "__main__":
    build_features()