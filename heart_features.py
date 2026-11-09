"""Feature engineering shared by train.py, the Gradio Space and the browser export.

Mirrors the notebook (sections 4-7): RestingBP zeros -> median, IQR capping of
MaxHR / RestingBP / Oldpeak / Cholesterol, medical binning of Oldpeak and
Cholesterol, RestingECG dropped, one-hot encoding of the categoricals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RAW_FEATURES = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS",
    "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope",
]
TARGET = "HeartDisease"
CAPPED = ["MaxHR", "RestingBP", "Oldpeak", "Cholesterol"]

OLDPEAK_BINS = [-np.inf, -0.01, 0.0, 1.0, 2.0, np.inf]
OLDPEAK_LABELS = ["ST Elevation", "Normal", "Mild Depression", "Moderate Depression", "Severe Depression"]
CHOL_BINS = [-np.inf, 200, 240, np.inf]
CHOL_LABELS = ["Normal", "Borderline", "High"]

CATEGORIES = {
    "Sex": ["F", "M"],
    "ChestPainType": ["ASY", "ATA", "NAP", "TA"],
    "ExerciseAngina": ["N", "Y"],
    "ST_Slope": ["Down", "Flat", "Up"],
    "Oldpeak": sorted(OLDPEAK_LABELS),
    "Cholesterol": sorted(CHOL_LABELS),
}
NUMERIC = ["Age", "RestingBP", "FastingBS", "MaxHR"]
FEATURE_NAMES = NUMERIC + [f"{c}_{v}" for c, vals in CATEGORIES.items() for v in vals]


def fit_params(df: pd.DataFrame) -> dict:
    """Learn the cleaning parameters (median for RestingBP zeros, IQR caps)."""
    df = df.copy()
    median_bp = float(df.loc[df["RestingBP"] > 0, "RestingBP"].median())
    df["RestingBP"] = df["RestingBP"].replace(0, median_bp)
    caps = {}
    for col in CAPPED:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        caps[col] = [float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)]
    return {"restingbp_median": median_bp, "caps": caps}


def transform(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Raw rows (the 11 Kaggle columns) -> the 23 model features."""
    df = df.copy()
    df["RestingBP"] = df["RestingBP"].replace(0, params["restingbp_median"])
    for col, (lo, hi) in params["caps"].items():
        df[col] = df[col].clip(lo, hi)
    df["Oldpeak"] = pd.cut(df["Oldpeak"], OLDPEAK_BINS, labels=OLDPEAK_LABELS).astype(str)
    df["Cholesterol"] = pd.cut(df["Cholesterol"], CHOL_BINS, labels=CHOL_LABELS).astype(str)
    out = df[NUMERIC].astype(float)
    for col, vals in CATEGORIES.items():
        for v in vals:
            out[f"{col}_{v}"] = (df[col] == v).astype(int)
    return out[FEATURE_NAMES]
