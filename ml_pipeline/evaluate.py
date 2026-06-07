"""
evaluate.py
-----------
Loads saved models and generates evaluation reports + confusion matrix plot.
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             classification_report, roc_auc_score)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

import sys
sys.path.insert(0, BASE_DIR)
from ml_pipeline.feature_engineering import engineer_creditcard, engineer_transactions
from sklearn.model_selection import train_test_split


def plot_confusion_matrix(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legit", "Fraud"],
                yticklabels=["Legit", "Fraud"])
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [plot] Saved confusion matrix → {save_path}")


def plot_roc_curve(y_true, y_prob, title, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2,
             label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [plot] Saved ROC curve → {save_path}")


def evaluate_creditcard():
    print("\n── Evaluating CreditCard model ──")
    df = pd.read_csv(os.path.join(PROC_DIR, "creditcard_clean.csv"))
    X, y, _ = engineer_creditcard(df)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    model  = joblib.load(os.path.join(MODEL_DIR, "cc_fraud_model.pkl"))

    X_test_sc = scaler.transform(X_test)
    y_pred    = model.predict(X_test_sc)
    y_prob    = model.predict_proba(X_test_sc)[:, 1]

    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
    print(f"  ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

    plot_confusion_matrix(y_test, y_pred, "CreditCard Confusion Matrix",
                          os.path.join(REPORT_DIR, "cc_confusion_matrix.png"))
    plot_roc_curve(y_test, y_prob, "CreditCard ROC Curve",
                   os.path.join(REPORT_DIR, "cc_roc_curve.png"))


def evaluate_transactions():
    print("\n── Evaluating Transactions model ──")
    df = pd.read_csv(os.path.join(PROC_DIR, "transactions_clean.csv"))
    X, y, _ = engineer_transactions(df)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = joblib.load(os.path.join(MODEL_DIR, "txn_scaler.pkl"))
    model  = joblib.load(os.path.join(MODEL_DIR, "txn_fraud_model.pkl"))

    X_test_sc = scaler.transform(X_test)
    y_pred    = model.predict(X_test_sc)
    y_prob    = model.predict_proba(X_test_sc)[:, 1]

    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
    print(f"  ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

    plot_confusion_matrix(y_test, y_pred, "Transactions Confusion Matrix",
                          os.path.join(REPORT_DIR, "txn_confusion_matrix.png"))
    plot_roc_curve(y_test, y_prob, "Transactions ROC Curve",
                   os.path.join(REPORT_DIR, "txn_roc_curve.png"))


if __name__ == "__main__":
    evaluate_creditcard()
    evaluate_transactions()
    print("\n✅ Evaluation complete — check reports/")
