from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_dataset(rows: int = 1500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(21, 60, size=rows)
    annual_income = rng.normal(650000, 220000, size=rows).clip(180000, 1800000)
    loan_amount = rng.normal(300000, 140000, size=rows).clip(50000, 1000000)
    credit_score = rng.normal(690, 70, size=rows).clip(300, 850)
    employment_years = rng.integers(0, 20, size=rows)
    existing_loans = rng.integers(0, 5, size=rows)
    dependents = rng.integers(0, 5, size=rows)
    missed_payments = rng.integers(0, 7, size=rows)

    education = rng.choice(
        ["Graduate", "Postgraduate", "High School", "Diploma"],
        size=rows,
        p=[0.36, 0.16, 0.28, 0.20],
    )
    employment_type = rng.choice(
        ["Salaried", "Self-Employed", "Business", "Unemployed"],
        size=rows,
        p=[0.48, 0.22, 0.20, 0.10],
    )
    residence_type = rng.choice(
        ["Owned", "Rented", "Mortgaged"],
        size=rows,
        p=[0.42, 0.40, 0.18],
    )
    marital_status = rng.choice(
        ["Single", "Married", "Divorced"],
        size=rows,
        p=[0.42, 0.46, 0.12],
    )
    loan_purpose = rng.choice(
        ["Home", "Vehicle", "Education", "Business", "Personal"],
        size=rows,
        p=[0.22, 0.18, 0.15, 0.20, 0.25],
    )

    debt_to_income_ratio = ((loan_amount / annual_income) * 100).clip(3, 95)

    risk_score = (
        0.022 * (650 - credit_score)
        + 0.038 * debt_to_income_ratio
        + 0.55 * missed_payments
        + 0.24 * existing_loans
        + 0.035 * np.maximum(0, 3 - employment_years)
        + np.where(employment_type == "Unemployed", 1.6, 0)
        + np.where(employment_type == "Self-Employed", 0.35, 0)
        + np.where(residence_type == "Rented", 0.25, 0)
        + np.where(loan_purpose == "Personal", 0.22, 0)
        + np.where(loan_purpose == "Business", 0.18, 0)
        - 0.0000012 * annual_income
    )

    default_probability = 1 / (1 + np.exp(-(risk_score - 1.4)))
    default = rng.binomial(1, default_probability)

    dataset = pd.DataFrame(
        {
            "age": age,
            "annual_income": annual_income.round(0).astype(int),
            "loan_amount": loan_amount.round(0).astype(int),
            "credit_score": credit_score.round(0).astype(int),
            "employment_years": employment_years,
            "existing_loans": existing_loans,
            "dependents": dependents,
            "missed_payments": missed_payments,
            "debt_to_income_ratio": debt_to_income_ratio.round(2),
            "education": education,
            "employment_type": employment_type,
            "residence_type": residence_type,
            "marital_status": marital_status,
            "loan_purpose": loan_purpose,
            "default": default,
        }
    )
    return dataset


def save_dataset(output_path: str | Path = "data/loan_default_dataset.csv", rows: int = 1500) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    generate_dataset(rows=rows).to_csv(path, index=False)
    return path


if __name__ == "__main__":
    saved_path = save_dataset()
    print(f"Dataset saved to {saved_path}")
