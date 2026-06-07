"""
feature_engineering.py
-----------------------
Builds final feature sets for model training from cleaned data.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR  = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")


# ── Transaction (PaySim) features ──────────────────────────────────────────

TXN_FEATURES = [
    "step", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "errorBalanceOrig", "errorBalanceDest",
    "type_is_transfer", "type_is_cashout",
    "hour", "night_transaction",
    "zero_orig_after", "zero_dest_before",
    "type_encoded",
    "avg_credit_score", "avg_balance", "avg_age", "avg_salary",
    "amount_to_avg_balance",
]
TXN_TARGET = "isFraud"


# ── Credit Card (PCA) features ─────────────────────────────────────────────

CC_FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
CC_TARGET   = "Class"


def engineer_transactions(df: pd.DataFrame):
    df = df.copy()

    # Transaction velocity — rolling is expensive on 50k rows; use log-amount instead
    df["log_amount"] = np.log1p(df["amount"])
    df["log_oldbalanceOrg"] = np.log1p(df["oldbalanceOrg"])
    df["log_newbalanceDest"] = np.log1p(df["newbalanceDest"])

    # Ratio features
    df["balance_change_orig"] = (df["oldbalanceOrg"] - df["newbalanceOrig"]) / (df["oldbalanceOrg"] + 1)
    df["balance_change_dest"] = (df["newbalanceDest"] - df["oldbalanceDest"]) / (df["oldbalanceDest"] + 1)

    features = TXN_FEATURES + [
        "log_amount", "log_oldbalanceOrg", "log_newbalanceDest",
        "balance_change_orig", "balance_change_dest",
    ]
    # Keep only columns that exist
    features = [c for c in features if c in df.columns]

    X = df[features]
    y = df[TXN_TARGET]
    return X, y, features


def engineer_creditcard(df: pd.DataFrame):
    df = df.copy()
    df["log_amount"] = np.log1p(df["Amount"])
    df["hour_of_day"] = (df["Time"] % (3600 * 24)) / 3600  # normalise Time to 0–24

    features = CC_FEATURES + ["log_amount", "hour_of_day"]
    features = [c for c in features if c in df.columns]

    X = df[features]
    y = df[CC_TARGET]
    return X, y, features


def scale_and_save(X_train, X_test, scaler_path):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    joblib.dump(scaler, scaler_path)
    print(f"[scaler] saved → {scaler_path}")
    return X_train_scaled, X_test_scaled, scaler


if __name__ == "__main__":
    txn_df = pd.read_csv(os.path.join(PROC_DIR, "transactions_clean.csv"))
    cc_df  = pd.read_csv(os.path.join(PROC_DIR, "creditcard_clean.csv"))

    X_txn, y_txn, txn_feats = engineer_transactions(txn_df)
    X_cc,  y_cc,  cc_feats  = engineer_creditcard(cc_df)

    print(f"[txn] X:{X_txn.shape}  y fraud rate:{y_txn.mean():.4f}")
    print(f"[cc]  X:{X_cc.shape}   y fraud rate:{y_cc.mean():.4f}")
