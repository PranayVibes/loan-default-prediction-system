from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.train_model import FEATURE_COLUMNS, train_and_save


MODEL_PATH = Path("artifacts/loan_default_model.joblib")


def load_model():
    if not MODEL_PATH.exists():
        train_and_save()
    return joblib.load(MODEL_PATH)


def risk_label(probability: float) -> str:
    if probability < 0.3:
        return "Low Risk"
    if probability < 0.6:
        return "Medium Risk"
    return "High Risk"


def decision_label(probability: float) -> str:
    if probability < 0.3:
        return "Approve"
    if probability < 0.6:
        return "Review"
    return "Reject"


def predict_default_risk(input_df: pd.DataFrame) -> pd.DataFrame:
    model = load_model()
    prepared_df = input_df.copy()[FEATURE_COLUMNS]

    probability = model.predict_proba(prepared_df)[:, 1]
    prediction = (probability >= 0.5).astype(int)

    result = prepared_df.copy()
    result["default_probability"] = probability.round(4)
    result["predicted_default"] = prediction
    result["risk_level"] = [risk_label(p) for p in probability]
    result["decision"] = [decision_label(p) for p in probability]
    return result
