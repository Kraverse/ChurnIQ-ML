from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

app = FastAPI(title="ChurnIQ-ML API", version="0.1.0")
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "churn_model.joblib"
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


class CustomerInput(BaseModel):
    model_config = ConfigDict(extra="allow")


@app.get("/health")
def health():
    return {"status": "ok", "model_available": model is not None}


@app.post("/predict")
def predict(customer: CustomerInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model has not been trained yet")

    frame = pd.DataFrame([customer.model_dump()])
    try:
        prediction = int(model.predict(frame)[0])
        probability = float(model.predict_proba(frame)[0, 1])
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid customer features: {exc}",
        ) from exc

    return {
        "churn_prediction": prediction,
        "churn_probability": round(probability, 4),
    }
