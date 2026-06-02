from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_generator import save_dataset


NUMERIC_FEATURES = [
    "age",
    "annual_income",
    "loan_amount",
    "credit_score",
    "employment_years",
    "existing_loans",
    "dependents",
    "missed_payments",
    "debt_to_income_ratio",
]

CATEGORICAL_FEATURES = [
    "education",
    "employment_type",
    "residence_type",
    "marital_status",
    "loan_purpose",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "default"


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_lr_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )


def build_rf_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=12,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_and_save(
    data_path: str | Path = "data/loan_default_dataset.csv",
    model_path: str | Path = "artifacts/loan_default_model.joblib",
    metrics_path: str | Path = "artifacts/model_metrics.json",
) -> dict:
    data_path = Path(data_path)
    if not data_path.exists():
        save_dataset(data_path)

    df = pd.read_csv(data_path)
    x = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train both models
    lr_model = build_lr_pipeline()
    lr_model.fit(x_train, y_train)
    lr_pred = lr_model.predict(x_test)
    lr_prob = lr_model.predict_proba(x_test)[:, 1]
    lr_acc = round(float(accuracy_score(y_test, lr_pred)), 4)
    lr_auc = round(float(roc_auc_score(y_test, lr_prob)), 4)

    rf_model = build_rf_pipeline()
    rf_model.fit(x_train, y_train)
    rf_pred = rf_model.predict(x_test)
    rf_prob = rf_model.predict_proba(x_test)[:, 1]
    rf_acc = round(float(accuracy_score(y_test, rf_pred)), 4)
    rf_auc = round(float(roc_auc_score(y_test, rf_prob)), 4)

    # Pick best model by ROC-AUC
    if rf_auc >= lr_auc:
        best_model = rf_model
        best_pred = rf_pred
        best_prob = rf_prob
        best_name = "Random Forest"
    else:
        best_model = lr_model
        best_pred = lr_pred
        best_prob = lr_prob
        best_name = "Logistic Regression"

    cm = confusion_matrix(y_test, best_pred)
    fpr, tpr, _ = roc_curve(y_test, best_prob)

    metrics = {
        "accuracy": rf_acc if best_name == "Random Forest" else lr_acc,
        "roc_auc": rf_auc if best_name == "Random Forest" else lr_auc,
        "best_model": best_name,
        "report": classification_report(y_test, best_pred, output_dict=True),
        "features": FEATURE_COLUMNS,
        "comparison": {
            "Logistic Regression": {"accuracy": lr_acc, "roc_auc": lr_auc},
            "Random Forest": {"accuracy": rf_acc, "roc_auc": rf_auc},
        },
        "confusion_matrix": cm.tolist(),
        "roc_fpr": fpr.tolist(),
        "roc_tpr": tpr.tolist(),
        "lr_accuracy": lr_acc,
        "lr_roc_auc": lr_auc,
        "rf_accuracy": rf_acc,
        "rf_roc_auc": rf_auc,
    }

    model_path = Path(model_path)
    metrics_path = Path(metrics_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    return metrics


if __name__ == "__main__":
    scores = train_and_save()
    print(
        json.dumps(
            {
                "best_model": scores["best_model"],
                "accuracy": scores["accuracy"],
                "roc_auc": scores["roc_auc"],
            },
            indent=2,
        )
    )
