"""
preprocess.py
-------------
Loads raw CSVs, cleans them, merges relevant features,
and saves a single cleaned_data.csv to data/processed/.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR  = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def load_raw_data():
    transactions = pd.read_csv(os.path.join(RAW_DIR, "transactions.csv"))
    customers    = pd.read_csv(os.path.join(RAW_DIR, "customers.csv"))
    creditcard   = pd.read_csv(os.path.join(RAW_DIR, "creditcard.csv"))
    print(f"[load] transactions: {transactions.shape}, customers: {customers.shape}, creditcard: {creditcard.shape}")
    return transactions, customers, creditcard


def preprocess_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and engineer features from the PaySim transactions dataset."""
    df = df.copy()

    # Drop isFlaggedFraud (internal rule-based flag, not our target)
    df.drop(columns=["isFlaggedFraud"], inplace=True, errors="ignore")

    # Balance error features (strong fraud signal in PaySim)
    df["errorBalanceOrig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
    df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]

    # Only TRANSFER and CASH_OUT carry fraud in this dataset
    df["type_is_transfer"] = (df["type"] == "TRANSFER").astype(int)
    df["type_is_cashout"]  = (df["type"] == "CASH_OUT").astype(int)

    # Hour of day proxy from step (1 step ≈ 1 hour)
    df["hour"] = df["step"] % 24
    df["night_transaction"] = (df["hour"].between(0, 6)).astype(int)

    # Zero-balance origin (common in fraud)
    df["zero_orig_after"] = (df["newbalanceOrig"] == 0).astype(int)
    df["zero_dest_before"] = (df["oldbalanceDest"] == 0).astype(int)

    # Encode transaction type
    le_type = LabelEncoder()
    df["type_encoded"] = le_type.fit_transform(df["type"])

    # Drop raw string columns
    df.drop(columns=["nameOrig", "nameDest", "type"], inplace=True, errors="ignore")

    print(f"[preprocess_transactions] shape: {df.shape}, fraud rate: {df['isFraud'].mean():.4f}")
    return df, le_type


def preprocess_customers(df: pd.DataFrame) -> pd.DataFrame:
    """Clean customer churn/behaviour dataset to extract useful aggregate features."""
    df = df.copy()
    df.drop(columns=["RowNumber", "CustomerId", "Surname"], inplace=True, errors="ignore")

    le_geo    = LabelEncoder()
    le_gender = LabelEncoder()
    df["Geography_enc"] = le_geo.fit_transform(df["Geography"])
    df["Gender_enc"]    = le_gender.fit_transform(df["Gender"])
    df.drop(columns=["Geography", "Gender"], inplace=True)

    # Aggregate stats to simulate customer-level risk features
    customer_stats = {
        "avg_credit_score":  df["CreditScore"].mean(),
        "avg_balance":       df["Balance"].mean(),
        "avg_age":           df["Age"].mean(),
        "avg_salary":        df["EstimatedSalary"].mean(),
    }
    print(f"[preprocess_customers] customer stats computed")
    return customer_stats


def preprocess_creditcard(df: pd.DataFrame) -> pd.DataFrame:
    """The PCA-anonymised credit card dataset — primary fraud dataset."""
    df = df.copy()

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # No missing values expected but guard anyway
    df.dropna(inplace=True)

    print(f"[preprocess_creditcard] shape: {df.shape}, fraud rate: {df['Class'].mean():.4f}")
    return df


def build_enriched_transactions(transactions_df, customer_stats):
    """
    Enrich transaction-level data with global customer behaviour statistics.
    This simulates a join that would happen in a real DB.
    """
    df = transactions_df.copy()
    df["avg_credit_score"] = customer_stats["avg_credit_score"]
    df["avg_balance"]      = customer_stats["avg_balance"]
    df["avg_age"]          = customer_stats["avg_age"]
    df["avg_salary"]       = customer_stats["avg_salary"]

    # Relative amount compared to average customer balance
    df["amount_to_avg_balance"] = df["amount"] / (customer_stats["avg_balance"] + 1)
    return df


def run_preprocessing():
    transactions_raw, customers_raw, creditcard_raw = load_raw_data()

    # ---------- Transactions (PaySim) ----------
    transactions_clean, le_type = preprocess_transactions(transactions_raw)

    # ---------- Customers ----------
    customer_stats = preprocess_customers(customers_raw)

    # ---------- Enrich transactions ----------
    transactions_enriched = build_enriched_transactions(transactions_clean, customer_stats)

    # ---------- Credit Card ----------
    creditcard_clean = preprocess_creditcard(creditcard_raw)

    # ---------- Save ----------
    transactions_enriched.to_csv(os.path.join(PROC_DIR, "transactions_clean.csv"), index=False)
    creditcard_clean.to_csv(os.path.join(PROC_DIR, "creditcard_clean.csv"), index=False)

    # Save label encoder
    joblib.dump(le_type, os.path.join(MODEL_DIR, "label_encoders.pkl"))

    print("\n[preprocess] All files saved to data/processed/")
    print(f"  transactions_clean.csv : {transactions_enriched.shape}")
    print(f"  creditcard_clean.csv   : {creditcard_clean.shape}")

    return transactions_enriched, creditcard_clean


if __name__ == "__main__":
    run_preprocessing()
