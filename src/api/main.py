from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict_risk


app = FastAPI(
    title="CardioGuard API",
    description="API for cardiometabolic high-risk prediction",
    version="1.0.0"
)


class PatientData(BaseModel):
    age: float
    gender: str
    ethnicity: str
    education_level: str
    poverty_income_ratio: float

    age_65_plus_flag: int
    low_income_flag: int
    low_education_flag: int

    bmi: float
    waist_circumference: float
    weight: float
    height: float
    waist_to_height_ratio: float
    bmi_to_waist_ratio: float

    systolic_bp: float
    diastolic_bp: float
    pulse_pressure: float
    mean_arterial_pressure: float

    fasting_glucose: float
    hba1c: float
    glucose_hba1c_ratio: float

    total_cholesterol: float
    hdl_cholesterol: float
    ldl_cholesterol: float
    triglycerides: float
    tc_hdl_ratio: float
    ldl_hdl_ratio: float
    trig_hdl_ratio: float

    smoker_flag: int
    physically_active_flag: int
    inactive_flag: int


@app.get("/")
def home():
    return {
        "message": "Welcome to CardioGuard API",
        "status": "running"
    }


@app.post("/predict")
def predict(data: PatientData):
    input_data = data.dict()
    result = predict_risk(input_data)
    return result