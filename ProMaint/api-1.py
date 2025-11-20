from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import json

# ------------------------------
# Load model and feature order
# ------------------------------
MODEL_PATH = "model_rf_pipeline.joblib"
COLS_PATH = "cols.json"

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise Exception(f"Cannot load model: {e}")

try:
    with open(COLS_PATH, "r") as f:
        FEATURES_ORDER = json.load(f)
except Exception as e:
    raise Exception(f"Cannot load feature columns: {e}")

# ------------------------------
# FastAPI app instance
# ------------------------------
app = FastAPI(
    title="Predictive Maintenance API",
    description="Enter sensor readings and get human-readable maintenance advice.",
    version="1.0"
)

# ------------------------------
# Input schema using Pydantic
# ------------------------------
class SensorData(BaseModel):
    # dynamically add fields from FEATURES_ORDER
    def __init__(self, **data):
        super().__init__(**data)
        for col in FEATURES_ORDER:
            if col not in data:
                setattr(self, col, 0)

# ------------------------------
# Prediction endpoint
# ------------------------------
@app.post("/predict")
def predict(data: dict):
    try:
        # Convert input to the correct order
        values = [float(data.get(col, 0)) for col in FEATURES_ORDER]
        X = np.array([values])
        pred = int(model.predict(X)[0])
        prob = float(model.predict_proba(X)[0, 1]) if hasattr(model, "predict_proba") else None

        if pred == 1:
            message = (
                "Warning: The machine is prone to failure.\n"
                "Recommendation: Have the machine checked immediately. "
                "Check temperature, rotation speed, and instrument condition."
            )
        else:
            message = "The machine's condition appears stable now. Monitor periodically."

        response = {
            "prediction": pred,
            "probability": round(prob, 4) if prob is not None else None,
            "message": message
        }
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing input: {e}")
