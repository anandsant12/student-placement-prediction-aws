from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib


# Create FastAPI application
app = FastAPI(
    title="Student Placement Prediction API",
    description="ML API for predicting student placement",
    version="1.0.0"
)


# Request schema
class StudentRequest(BaseModel):
    iq: int
    city: str
    percentage: float


# Load trained model
model = joblib.load(
    "student_placement_model.joblib"
)


# Health/root endpoint
@app.get("/")
def home():
    return {
        "message": "Student Placement Prediction API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }




# Prediction endpoint
@app.post("/predict")
def predict(student: StudentRequest):

    # Convert request into DataFrame
    input_data = pd.DataFrame([
        {
            "iq": student.iq,
            "city": student.city,
            "percentage": student.percentage
        }
    ])

    # Make prediction
    prediction = model.predict(input_data)

    # Get probabilities
    probability = model.predict_proba(input_data)

    return {
        "placement": prediction[0],
        "probability_no": float(probability[0][0]),
        "probability_yes": float(probability[0][1])
    }