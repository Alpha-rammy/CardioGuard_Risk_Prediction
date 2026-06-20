import os
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(DATA_PROCESSED, exist_ok=True)


def standardize_columns(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df


def load_raw_data():
    return {
        "demo": pd.read_sas(os.path.join(DATA_RAW, "DEMO_J.XPT")),
        "bmx": pd.read_sas(os.path.join(DATA_RAW, "BMX_J.XPT")),
        "bpx": pd.read_sas(os.path.join(DATA_RAW, "BPX_J.XPT")),
        "glu": pd.read_sas(os.path.join(DATA_RAW, "GLU_J.XPT")),
        "ghb": pd.read_sas(os.path.join(DATA_RAW, "GHB_J.XPT")),
        "tchol": pd.read_sas(os.path.join(DATA_RAW, "TCHOL_J.XPT")),
        "hdl": pd.read_sas(os.path.join(DATA_RAW, "HDL_J.XPT")),
        "trigly": pd.read_sas(os.path.join(DATA_RAW, "TRIGLY_J.XPT")),
        "smq": pd.read_sas(os.path.join(DATA_RAW, "SMQ_J.XPT")),
        "paq": pd.read_sas(os.path.join(DATA_RAW, "PAQ_J.XPT")),
        "bpq": pd.read_sas(os.path.join(DATA_RAW, "BPQ_J.XPT")),
        "diq": pd.read_sas(os.path.join(DATA_RAW, "DIQ_J.XPT")),
    }


def preprocess_data():
    data = load_raw_data()

    for name in data:
        data[name] = standardize_columns(data[name])

    demo = data["demo"][["seqn", "riagendr", "ridageyr", "ridreth3", "dmdeduc2", "indfmpir"]]
    bmx = data["bmx"][["seqn", "bmxbmi", "bmxwaist", "bmxwt", "bmxht"]]
    bpx = data["bpx"][["seqn", "bpxsy1", "bpxdi1"]]
    glu = data["glu"][["seqn", "lbxglu"]]
    ghb = data["ghb"][["seqn", "lbxgh"]]
    tchol = data["tchol"][["seqn", "lbxtc"]]
    hdl = data["hdl"][["seqn", "lbdhdd"]]
    trigly = data["trigly"][["seqn", "lbxtr", "lbdldl"]]
    smq = data["smq"][["seqn", "smq020"]]
    paq = data["paq"][["seqn", "paq650", "paq665"]]
    bpq = data["bpq"][["seqn", "bpq020", "bpq040a"]]
    diq = data["diq"][["seqn", "diq010"]]

    datasets = [
        demo, bmx, bpx, glu, ghb,
        tchol, hdl, trigly,
        smq, paq, bpq, diq
    ]

    for df in datasets:
        df.rename(columns={"seqn": "participant_id"}, inplace=True)

    demo.rename(columns={
        "riagendr": "gender",
        "ridageyr": "age",
        "ridreth3": "ethnicity",
        "dmdeduc2": "education_level",
        "indfmpir": "poverty_income_ratio"
    }, inplace=True)

    bmx.rename(columns={
        "bmxbmi": "bmi",
        "bmxwaist": "waist_circumference",
        "bmxwt": "weight",
        "bmxht": "height"
    }, inplace=True)

    bpx.rename(columns={
        "bpxsy1": "systolic_bp",
        "bpxdi1": "diastolic_bp"
    }, inplace=True)

    glu.rename(columns={"lbxglu": "fasting_glucose"}, inplace=True)
    ghb.rename(columns={"lbxgh": "hba1c"}, inplace=True)
    tchol.rename(columns={"lbxtc": "total_cholesterol"}, inplace=True)
    hdl.rename(columns={"lbdhdd": "hdl_cholesterol"}, inplace=True)

    trigly.rename(columns={
        "lbxtr": "triglycerides",
        "lbdldl": "ldl_cholesterol"
    }, inplace=True)

    smq.rename(columns={"smq020": "smoking_status"}, inplace=True)

    paq.rename(columns={
        "paq650": "vigorous_activity",
        "paq665": "moderate_activity"
    }, inplace=True)

    bpq.rename(columns={
        "bpq020": "hypertension_history",
        "bpq040a": "bp_medication"
    }, inplace=True)

    diq.rename(columns={"diq010": "diabetes_history"}, inplace=True)

    gender_map = {1: "Male", 2: "Female"}

    ethnicity_map = {
        1: "Mexican American",
        2: "Other Hispanic",
        3: "Non-Hispanic White",
        4: "Non-Hispanic Black",
        6: "Non-Hispanic Asian",
        7: "Other Race"
    }

    education_map = {
        1: "Less than 9th grade",
        2: "9th-11th grade",
        3: "High school graduate",
        4: "Some college",
        5: "College graduate or above",
        7: np.nan,
        9: np.nan
    }

    yes_no_map = {
        1: "Yes",
        2: "No",
        7: np.nan,
        9: np.nan
    }

    diabetes_map = {
        1: "Yes",
        2: "No",
        3: "Borderline",
        7: np.nan,
        9: np.nan
    }

    demo["gender"] = demo["gender"].map(gender_map)
    demo["ethnicity"] = demo["ethnicity"].map(ethnicity_map)
    demo["education_level"] = demo["education_level"].map(education_map)

    smq["smoking_status"] = smq["smoking_status"].map(yes_no_map)

    paq["vigorous_activity"] = paq["vigorous_activity"].map(yes_no_map)
    paq["moderate_activity"] = paq["moderate_activity"].map(yes_no_map)

    bpq["hypertension_history"] = bpq["hypertension_history"].map(yes_no_map)
    bpq["bp_medication"] = bpq["bp_medication"].map(yes_no_map)

    diq["diabetes_history"] = diq["diabetes_history"].map(diabetes_map)

    numeric_datasets = [demo, bmx, bpx, glu, ghb, tchol, hdl, trigly]

    for df in numeric_datasets:
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

        for col in numeric_cols:
            if col != "participant_id":
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].clip(lower=0)

    cleaned_files = {
        "demo_clean.csv": demo,
        "bmx_clean.csv": bmx,
        "bpx_clean.csv": bpx,
        "glu_clean.csv": glu,
        "ghb_clean.csv": ghb,
        "tchol_clean.csv": tchol,
        "hdl_clean.csv": hdl,
        "trigly_clean.csv": trigly,
        "smq_clean.csv": smq,
        "paq_clean.csv": paq,
        "bpq_clean.csv": bpq,
        "diq_clean.csv": diq
    }

    for filename, df in cleaned_files.items():
        df.drop_duplicates(inplace=True)

        output_path = os.path.join(DATA_PROCESSED, filename)
        df.to_csv(output_path, index=False)

        print(f"{filename} saved successfully")

    print("Preprocessing completed.")


if __name__ == "__main__":
    preprocess_data()