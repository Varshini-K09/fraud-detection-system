"""
train.py
--------
Trains Logistic Regression, Random Forest, and XGBoost models on both
the PaySim transactions dataset and the Credit Card fraud dataset.
Saves the best model per dataset.
"""

import os, json, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR  = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Import local modules
import sys
sys.path.insert(0, BASE_DIR)
from ml_pipeline.feature_engineering import (
    engineer_transactions, engineer_creditcard, scale_and_save
)


# ── Helpers ───────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, name=""):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    report = {
        "model":     name,
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
        "f1":        round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
    }
    print(f"\n  [{name}] AUC={report['roc_auc']} | F1={report['f1']} "
          f"| Precision={report['precision']} | Recall={report['recall']}")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
    return report, y_pred


def apply_smote(X_train, y_train, sampling_strategy=0.1, random_state=42):
    """Apply SMOTE only when minority class is very small."""
    fraud_ratio = y_train.mean()
    print(f"  [smote] fraud ratio before: {fraud_ratio:.4f}")
    if fraud_ratio < 0.05:
        sm = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        print(f"  [smote] after resampling: {y_res.mean():.4f} | shape: {X_res.shape}")
        return X_res, y_res
    return X_train, y_train


# ── Train on a dataset ────────────────────────────────────────────────────

def train_dataset(X, y, dataset_name, scaler_name, model_name_prefix):
    print(f"\n{'='*60}")
    print(f" Training on: {dataset_name}  | shape: {X.shape} | fraud: {y.mean():.4f}")
    print(f"{'='*60}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Scale
    scaler_path = os.path.join(MODEL_DIR, scaler_name)
    X_train_sc, X_test_sc, _ = scale_and_save(X_train, X_test, scaler_path)

    # SMOTE
    X_res, y_res = apply_smote(X_train_sc, y_train)

    # Compute class weights for models that support it
    classes = np.unique(y_train)
    cw = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, cw))

    # ── Model Zoo ─────────────────────────────────────────────────────────
    models = {
        "LogisticRegression": LogisticRegression(
            C=0.1, max_iter=1000, class_weight="balanced", random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=int((y_res == 0).sum() / max((y_res == 1).sum(), 1)),
            eval_metric="aucpr", use_label_encoder=False,
            random_state=42, n_jobs=-1, verbosity=0
        ),
    }

    results = []
    trained = {}

    for mname, model in models.items():
        print(f"\n  ▶ Training {mname} ...")
        model.fit(X_res, y_res)
        report, _ = evaluate_model(model, X_test_sc, y_test, name=mname)
        results.append(report)
        trained[mname] = model

    # Best model by F1
    best_report = max(results, key=lambda r: r["f1"])
    best_model  = trained[best_report["model"]]
    print(f"\n  ✅ Best model: {best_report['model']}  (F1={best_report['f1']})")

    # Save best model
    model_path = os.path.join(MODEL_DIR, f"{model_name_prefix}_fraud_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"  💾 Saved → {model_path}")

    # Save results
    results_path = os.path.join(REPORT_DIR, f"{model_name_prefix}_results.json")
    with open(results_path, "w") as f:
        json.dump({"dataset": dataset_name, "models": results, "best": best_report}, f, indent=2)

    return best_model, best_report, X_test_sc, y_test


# ── Main ──────────────────────────────────────────────────────────────────

def run_training():
    # ------ Credit Card dataset (primary model) ------
    cc_df = pd.read_csv(os.path.join(PROC_DIR, "creditcard_clean.csv"))
    X_cc, y_cc, cc_feats = engineer_creditcard(cc_df)
    joblib.dump(cc_feats, os.path.join(MODEL_DIR, "cc_features.pkl"))

    cc_model, cc_report, X_cc_test, y_cc_test = train_dataset(
        X_cc, y_cc,
        dataset_name="CreditCard",
        scaler_name="scaler.pkl",          # primary scaler
        model_name_prefix="cc",
    )

    # ------ Transactions dataset (secondary model) ------
    txn_df = pd.read_csv(os.path.join(PROC_DIR, "transactions_clean.csv"))
    X_txn, y_txn, txn_feats = engineer_transactions(txn_df)
    joblib.dump(txn_feats, os.path.join(MODEL_DIR, "txn_features.pkl"))

    txn_model, txn_report, X_txn_test, y_txn_test = train_dataset(
        X_txn, y_txn,
        dataset_name="Transactions",
        scaler_name="txn_scaler.pkl",
        model_name_prefix="txn",
    )

    # Also save combined summary
    summary = {
        "creditcard": cc_report,
        "transactions": txn_report,
    }
    with open(os.path.join(REPORT_DIR, "model_results.txt"), "w") as f:
        f.write("=== Fraud Detection Model Results ===\n\n")
        for ds, r in summary.items():
            f.write(f"Dataset  : {ds}\n")
            f.write(f"Best Model: {r['model']}\n")
            f.write(f"ROC-AUC  : {r['roc_auc']}\n")
            f.write(f"F1-Score : {r['f1']}\n")
            f.write(f"Precision: {r['precision']}\n")
            f.write(f"Recall   : {r['recall']}\n")
            f.write("\n")

    print("\n\n✅ Training complete. Models saved to /models/")
    print(f"   CreditCard  → AUC={cc_report['roc_auc']}  F1={cc_report['f1']}")
    print(f"   Transactions→ AUC={txn_report['roc_auc']} F1={txn_report['f1']}")
    return summary


if __name__ == "__main__":
    run_training()
