"""
utils.py
--------
Shared utility functions used across the ML pipeline and backend.
"""

import os
import numpy as np
import pandas as pd
import joblib

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")


def load_artifacts(dataset="cc"):
    """Load model + scaler for a given dataset ('cc' or 'txn')."""
    if dataset == "cc":
        model  = joblib.load(os.path.join(MODEL_DIR, "cc_fraud_model.pkl"))
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        feats  = joblib.load(os.path.join(MODEL_DIR, "cc_features.pkl"))
    else:
        model  = joblib.load(os.path.join(MODEL_DIR, "txn_fraud_model.pkl"))
        scaler = joblib.load(os.path.join(MODEL_DIR, "txn_scaler.pkl"))
        feats  = joblib.load(os.path.join(MODEL_DIR, "txn_features.pkl"))
    return model, scaler, feats


def predict_fraud(input_dict: dict, dataset="cc") -> dict:
    """
    Make a fraud prediction given a flat input dictionary.
    Returns {'fraud_probability': float, 'is_fraud': bool, 'risk_level': str}
    """
    model, scaler, feats = load_artifacts(dataset)

    # Build feature row
    row = {f: input_dict.get(f, 0) for f in feats}
    X   = pd.DataFrame([row])[feats]
    X_sc = scaler.transform(X)

    prob      = float(model.predict_proba(X_sc)[0, 1])
    is_fraud  = prob >= 0.5

    if prob >= 0.8:
        risk = "HIGH"
    elif prob >= 0.5:
        risk = "MEDIUM"
    elif prob >= 0.3:
        risk = "LOW"
    else:
        risk = "SAFE"

    return {
        "fraud_probability": round(prob, 4),
        "is_fraud":          is_fraud,
        "risk_level":        risk,
    }


def describe_dataset_stats():
    """Return quick stats for dashboard use."""
    stats = {}
    for name, path in [
        ("creditcard", os.path.join(BASE_DIR, "data", "processed", "creditcard_clean.csv")),
        ("transactions", os.path.join(BASE_DIR, "data", "processed", "transactions_clean.csv")),
    ]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            target = "Class" if name == "creditcard" else "isFraud"
            stats[name] = {
                "total":       int(len(df)),
                "fraud_count": int(df[target].sum()),
                "fraud_rate":  round(float(df[target].mean()), 4),
            }
    return stats
