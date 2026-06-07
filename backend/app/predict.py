"""
predict.py
----------
Loads models once at startup and exposes prediction functions used by routes.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

MODEL_DIR = os.path.join(BASE_DIR, "models")

from ml_pipeline.feature_engineering import engineer_creditcard, engineer_transactions


# ── Lazy-load models once ──────────────────────────────────────────────────

_cc_model   = None
_cc_scaler  = None
_cc_feats   = None

_txn_model  = None
_txn_scaler = None
_txn_feats  = None


def _load_cc():
    global _cc_model, _cc_scaler, _cc_feats
    if _cc_model is None:
        _cc_model  = joblib.load(os.path.join(MODEL_DIR, "cc_fraud_model.pkl"))
        _cc_scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        _cc_feats  = joblib.load(os.path.join(MODEL_DIR, "cc_features.pkl"))
    return _cc_model, _cc_scaler, _cc_feats


def _load_txn():
    global _txn_model, _txn_scaler, _txn_feats
    if _txn_model is None:
        _txn_model  = joblib.load(os.path.join(MODEL_DIR, "txn_fraud_model.pkl"))
        _txn_scaler = joblib.load(os.path.join(MODEL_DIR, "txn_scaler.pkl"))
        _txn_feats  = joblib.load(os.path.join(MODEL_DIR, "txn_features.pkl"))
    return _txn_model, _txn_scaler, _txn_feats


def _risk_level(prob: float) -> str:
    if prob >= 0.8:  return "HIGH"
    if prob >= 0.5:  return "MEDIUM"
    if prob >= 0.3:  return "LOW"
    return "SAFE"


# ── Public API ─────────────────────────────────────────────────────────────

def predict_creditcard(input_data: dict) -> dict:
    model, scaler, feats = _load_cc()

    # Derived features
    import math
    row = dict(input_data)
    row["log_amount"]  = math.log1p(row.get("Amount", 0))
    row["hour_of_day"] = (row.get("Time", 0) % (3600 * 24)) / 3600

    X = pd.DataFrame([{f: row.get(f, 0) for f in feats}])[feats]
    X_sc = scaler.transform(X)

    prob = float(model.predict_proba(X_sc)[0, 1])
    return {
        "fraud_probability": round(prob, 4),
        "is_fraud":          prob >= 0.5,
        "risk_level":        _risk_level(prob),
    }


def predict_transaction(input_data: dict) -> dict:
    import math
    model, scaler, feats = _load_txn()

    row = dict(input_data)
    t = row.get("transaction_type", "PAYMENT")

    # Replicate feature engineering
    type_map = {"PAYMENT": 3, "TRANSFER": 4, "CASH_OUT": 1, "DEBIT": 2, "CASH_IN": 0}
    row["type_encoded"]        = type_map.get(t, 3)
    row["type_is_transfer"]    = int(t == "TRANSFER")
    row["type_is_cashout"]     = int(t == "CASH_OUT")
    row["hour"]                = row.get("step", 0) % 24
    row["night_transaction"]   = int(row["hour"] <= 6)
    row["zero_orig_after"]     = int(row.get("newbalanceOrig", 0) == 0)
    row["zero_dest_before"]    = int(row.get("oldbalanceDest", 0) == 0)

    orig  = row.get("oldbalanceOrg", 0)
    orig2 = row.get("newbalanceOrig", 0)
    dest1 = row.get("oldbalanceDest", 0)
    dest2 = row.get("newbalanceDest", 0)
    amt   = row.get("amount", 0)

    row["errorBalanceOrig"]  = orig2 + amt - orig
    row["errorBalanceDest"]  = dest1 + amt - dest2
    row["log_amount"]        = math.log1p(amt)
    row["log_oldbalanceOrg"] = math.log1p(orig)
    row["log_newbalanceDest"]= math.log1p(dest2)
    row["balance_change_orig"] = (orig - orig2) / (orig + 1)
    row["balance_change_dest"] = (dest2 - dest1) / (dest1 + 1)

    # Customer avg defaults (median from training)
    row.setdefault("avg_credit_score", 650.0)
    row.setdefault("avg_balance",      76485.89)
    row.setdefault("avg_age",          38.92)
    row.setdefault("avg_salary",       100090.24)
    row["amount_to_avg_balance"] = amt / (row["avg_balance"] + 1)

    X = pd.DataFrame([{f: row.get(f, 0) for f in feats}])[feats]
    X_sc = scaler.transform(X)

    prob = float(model.predict_proba(X_sc)[0, 1])
    return {
        "fraud_probability": round(prob, 4),
        "is_fraud":          prob >= 0.5,
        "risk_level":        _risk_level(prob),
    }
