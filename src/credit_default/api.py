
from functools import lru_cache
from pathlib import Path
import os
import sqlite3

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
DB_PATH = ROOT / "data" / "credit_default.db"
SCHEMA = ROOT / "sql" / "schema.sql"

app = FastAPI(
    title="Credit Default Prediction API",
    version="1.0.0",
    description=(
        "Research portfolio API. Not a lending decision system."
    ),
)

FEATURES = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
    "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]

class CreditFeatures(BaseModel):
    LIMIT_BAL: float = Field(ge=0)
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int = Field(ge=18, le=120)
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float = Field(ge=0)
    PAY_AMT2: float = Field(ge=0)
    PAY_AMT3: float = Field(ge=0)
    PAY_AMT4: float = Field(ge=0)
    PAY_AMT5: float = Field(ge=0)
    PAY_AMT6: float = Field(ge=0)

@lru_cache(maxsize=3)
def load_model(name: str):
    if name in {"logistic_regression", "random_forest"}:
        path = MODELS / f"{name}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Missing model: {path}")
        return joblib.load(path)

    if name == "neural_network":
        import tensorflow as tf

        model_path = MODELS / "neural_network.keras"
        prep_path = MODELS / "dl_preprocessor.joblib"

        if not model_path.exists() or not prep_path.exists():
            raise FileNotFoundError("Train the DL model first")

        return (
            tf.keras.models.load_model(model_path),
            joblib.load(prep_path),
        )

    raise ValueError("Unsupported model name")

@app.on_event("startup")
def initialize_database():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        con.executescript(SCHEMA.read_text())

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/models")
def list_models():
    return {
        "available_models": [
            "logistic_regression",
            "random_forest",
            "neural_network",
        ]
    }

@app.post("/predict")
def predict(
    features: CreditFeatures,
    model_name: str = "random_forest",
):
    row = pd.DataFrame(
        [features.model_dump()],
        columns=FEATURES,
    )

    try:
        if model_name == "neural_network":
            model, preprocessor = load_model(model_name)
            transformed = preprocessor.transform(row)
            import numpy as np
            probability = float(
                model.predict(
                    np.asarray(transformed, dtype="float32"),
                    verbose=0,
                )[0][0]
            )
        else:
            model = load_model(model_name)
            probability = float(
                model.predict_proba(row)[0, 1]
            )

    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    probability = min(max(probability, 0.0), 1.0)
    predicted = int(probability >= 0.5)

    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """INSERT INTO prediction_log
               (model_name, model_version,
                default_probability, predicted_default)
               VALUES (?, ?, ?, ?)""",
            (model_name, "v1", probability, predicted),
        )

    return {
        "model_name": model_name,
        "model_version": "v1",
        "default_probability": probability,
        "predicted_default_at_0_5": predicted,
        "threshold": 0.5,
        "notice": (
            "Research output only; not a lending decision."
        ),
    }
