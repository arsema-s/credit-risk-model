from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI

from src.api.pydantic_models import (
    CreditRiskRequest
)


app = FastAPI(
    title="Credit Risk API"
)


BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent.parent.parent
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "credit_risk_model.pkl"
)

model = joblib.load(
    MODEL_PATH
)


@app.get("/")
def home():

    return {
        "message":
        "Credit Risk API Running"
    }


@app.post("/predict")
def predict(
    request: CreditRiskRequest
):

    data = pd.DataFrame(
        [
            {
                "TotalTransactionAmount":
                request.TotalTransactionAmount,

                "AverageTransactionAmount":
                request.AverageTransactionAmount,

                "TransactionCount":
                request.TransactionCount,

                "StdTransactionAmount":
                request.StdTransactionAmount,

                "FraudResult":
                request.FraudResult
            }
        ]
    )

    prediction = (
        model.predict(data)[0]
    )

    probability = (
        model.predict_proba(data)[0][1]
    )

    return {
        "prediction":
        int(prediction),

        "probability":
        float(probability)
    }