from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import boto3
import os
import tempfile
from dotenv import load_dotenv


# Load environment variables from .env
load_dotenv()


app = FastAPI(
    title="Student Placement Prediction API - AWS",
    description="ML API for predicting student placement using model stored in AWS S3",
    version="1.0.0"
)


class StudentRequest(BaseModel):
    iq: int
    city: str
    percentage: float


# AWS / S3 configuration
S3_BUCKET = os.getenv("S3_BUCKET")
S3_KEY = os.getenv("S3_KEY")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION")
MODEL_VERSION = "v1"

def load_model_from_s3():

    print("Connecting to AWS S3...")

    s3 = boto3.client(
        "s3",
        region_name=AWS_REGION
    )

    print(f"Downloading model from s3://{S3_BUCKET}/{S3_KEY}")

    with tempfile.NamedTemporaryFile(suffix=".joblib") as temp_file:

        s3.download_file(
            S3_BUCKET,
            S3_KEY,
            temp_file.name
        )

        print("Model downloaded successfully.")

        model = joblib.load(temp_file.name)

        print("Model loaded successfully.")

        return model


# Load model when application starts
model = load_model_from_s3()


@app.get("/")
def home():
    return {
        "message": "Student Placement Prediction API using AWS S3"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_version": MODEL_VERSION
    }


@app.post("/predict")
def predict(student: StudentRequest):

    input_data = pd.DataFrame([
        {
            "iq": student.iq,
            "city": student.city,
            "percentage": student.percentage
        }
    ])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    return {
        "placement": prediction[0],
        "probability_no": float(probability[0][0]),
        "probability_yes": float(probability[0][1])
    }
