import streamlit as st
from src.predict import predict_risk

st.set_page_config(
    page_title="CardioGuard Risk Predictor",
    page_icon="🫀",
    layout="centered"
)

st.title("🫀 CardioGuard")
st.subheader("Cardiometabolic Risk Prediction App")

st.write(
    "Enter participant clinical and lifestyle information to estimate cardiometabolic high-risk probability."
)

age = st.number_input("Age", min_value=18, max_value=100, value=55)
gender = st.selectbox("Gender", ["Male", "Female"])
ethnicity = st.selectbox(
    "Ethnicity",
    [
        "Mexican American",
        "Other Hispanic",
        "Non-Hispanic White",
        "Non-Hispanic Black",
        "Non-Hispanic Asian",
        "Other Race"
    ]
)

education_level = st.selectbox(
    "Education Level",
    [
        "Less than 9th grade",
        "9th-11th grade",
        "High school graduate",
        "Some college",
        "College graduate or above"
    ]
)

poverty_income_ratio = st.number_input(
    "Poverty Income Ratio",
    min_value=0.0,
    max_value=10.0,
    value=1.8
)

bmi = st.number_input("BMI", min_value=10.0, max_value=80.0, value=32.5)
waist = st.number_input("Waist Circumference", min_value=40.0, max_value=200.0, value=105.0)
weight = st.number_input("Weight", min_value=20.0, max_value=250.0, value=92.0)
height = st.number_input("Height", min_value=80.0, max_value=230.0, value=175.0)

systolic_bp = st.number_input("Systolic BP", min_value=70.0, max_value=260.0, value=145.0)
diastolic_bp = st.number_input("Diastolic BP", min_value=40.0, max_value=160.0, value=92.0)

fasting_glucose = st.number_input("Fasting Glucose", min_value=40.0, max_value=500.0, value=130.0)
hba1c = st.number_input("HbA1c", min_value=3.0, max_value=15.0, value=6.8)

total_cholesterol = st.number_input("Total Cholesterol", min_value=50.0, max_value=500.0, value=220.0)
hdl_cholesterol = st.number_input("HDL Cholesterol", min_value=10.0, max_value=150.0, value=38.0)
ldl_cholesterol = st.number_input("LDL Cholesterol", min_value=10.0, max_value=300.0, value=145.0)
triglycerides = st.number_input("Triglycerides", min_value=20.0, max_value=1000.0, value=180.0)

smoker = st.selectbox("Smoking Status", ["No", "Yes"])
physically_active = st.selectbox("Physically Active", ["Yes", "No"])

if st.button("Predict Risk"):

    input_data = {
        "age": age,
        "gender": gender,
        "ethnicity": ethnicity,
        "education_level": education_level,
        "poverty_income_ratio": poverty_income_ratio,

        "age_65_plus_flag": int(age >= 65),
        "low_income_flag": int(poverty_income_ratio < 1),
        "low_education_flag": int(education_level in ["Less than 9th grade", "9th-11th grade"]),

        "bmi": bmi,
        "waist_circumference": waist,
        "weight": weight,
        "height": height,
        "waist_to_height_ratio": waist / height,
        "bmi_to_waist_ratio": bmi / waist,

        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "pulse_pressure": systolic_bp - diastolic_bp,
        "mean_arterial_pressure": ((2 * diastolic_bp) + systolic_bp) / 3,

        "fasting_glucose": fasting_glucose,
        "hba1c": hba1c,
        "glucose_hba1c_ratio": fasting_glucose / hba1c,

        "total_cholesterol": total_cholesterol,
        "hdl_cholesterol": hdl_cholesterol,
        "ldl_cholesterol": ldl_cholesterol,
        "triglycerides": triglycerides,
        "tc_hdl_ratio": total_cholesterol / hdl_cholesterol,
        "ldl_hdl_ratio": ldl_cholesterol / hdl_cholesterol,
        "trig_hdl_ratio": triglycerides / hdl_cholesterol,

        "smoker_flag": int(smoker == "Yes"),
        "physically_active_flag": int(physically_active == "Yes"),
        "inactive_flag": int(physically_active == "No")
    }

    result = predict_risk(input_data)

    st.subheader("Prediction Result")

    st.metric(
        "Risk Probability",
        f"{result['risk_probability'] * 100:.2f}%"
    )

    if result["risk_band"] == "High Risk":
        st.error(f"Risk Band: {result['risk_band']}")
    elif result["risk_band"] == "Moderate Risk":
        st.warning(f"Risk Band: {result['risk_band']}")
    else:
        st.success(f"Risk Band: {result['risk_band']}")

    st.write("Model Output:")
    st.json(result)