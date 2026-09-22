from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

app = FastAPI(title="ChurnIQ-ML API", version="0.1.0")
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "churn_model.joblib"


class CustomerInput(BaseModel):
    model_config = ConfigDict(extra="allow")


@app.get("/health")
def health():
    return {"status": "ok", "model_available": MODEL_PATH.exists()}


@app.post("/predict")
def predict(customer: CustomerInput):
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model has not been trained yet")
    model = joblib.load(MODEL_PATH)
    frame = pd.DataFrame([customer.model_dump()])
    try:
        prediction = int(model.predict(frame)[0])
        probability = float(model.predict_proba(frame)[0, 1])
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid customer features: {exc}") from exc
    return {"churn_prediction": prediction, "churn_probability": round(probability, 4)}
