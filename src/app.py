from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from utils import create_features

# Load model at startup
model = joblib.load("../models/xgb_model.joblib")


# Define expected input structure
class PredictionRequest(BaseModel):
    Open: float
    High: float
    Low: float
    Close: float
    Volume: float


app = FastAPI(title="Portfolio Prediction API")


@app.post("/api/predict")
def predict(request: PredictionRequest):
    input_df = pd.DataFrame([request.model_dump()])

    processed_df, feature_cols = create_features(input_df, for_simulation=True)

    expected_features = model.feature_names_in_
    X = processed_df[expected_features]

    pred_return = model.predict(X)[0]

    return {"predicted_return": float(pred_return)}

from typing import List

class BatchPredictionRequest(BaseModel):
    data: List[PredictionRequest]

@app.post("/api/batch_predict")
def batch_predict(request: BatchPredictionRequest):
    input_df = pd.DataFrame([row.model_dump() for row in request.data])

    processed_df, _ = create_features(input_df, for_simulation=True)

    expected_features = model.feature_names_in_
    X = processed_df[expected_features]

    preds = model.predict(X)

    return {"predicted_returns": preds.tolist()}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
