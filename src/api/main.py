from fastapi import FastAPI
from src.api.pydantic_models import PredictionRequest, PredictionResponse
import pandas as pd
import mlflow.pyfunc

app = FastAPI(title="Credit Risk Prediction API")

# Load the model from MLflow Registry
MODEL_NAME = "credit-risk-model"
MODEL_STAGE = "Production"

model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}")

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    input_df = pd.DataFrame([request.dict()])
    prediction = model.predict(input_df)
    return PredictionResponse(risk_probability=round(float(prediction[0]), 4))
