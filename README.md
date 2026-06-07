# 🛡️ Credit Card Fraud Detection System

End-to-end ML project for detecting fraudulent financial transactions.

---

## 📊 Model Results

| Dataset      | Model   | ROC-AUC | F1     | Recall | Precision |
|-------------|---------|---------|--------|--------|-----------|
| Credit Card | XGBoost | 0.9853  | 0.6039 | 0.8105 | 0.4813    |
| Transactions| XGBoost | 0.9392  | 0.8000 | 0.8333 | 0.7692    |

> **Recall is prioritised** — missing a real fraud is far more costly than a false alert.

---

## 🗂 Project Structure

```
fraud-detection-system/
├── data/
│   ├── raw/              ← Original CSVs (transactions, customers, creditcard)
│   └── processed/        ← Cleaned data (auto-generated)
├── notebooks/            ← EDA, feature engineering, model training notebooks
├── ml_pipeline/
│   ├── preprocess.py     ← Data cleaning & enrichment
│   ├── feature_engineering.py
│   ├── train.py          ← Model training (LR, RF, XGBoost)
│   ├── evaluate.py       ← Metrics & plots
│   └── utils.py          ← Shared helpers
├── models/               ← Saved .pkl files (auto-generated)
├── backend/app/
│   ├── main.py           ← FastAPI app
│   ├── predict.py        ← Model inference
│   ├── database.py       ← SQLAlchemy setup
│   ├── models.py         ← ORM models
│   ├── schemas.py        ← Pydantic schemas
│   └── routes/fraud_routes.py
├── frontend/             ← Dashboard (HTML/CSS/JS)
├── database/schema.sql   ← PostgreSQL schema
├── reports/              ← Confusion matrices, ROC curves
└── tests/test_api.py
```

---

## 🚀 Step-by-Step Setup

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Place raw CSV files in `data/raw/`
```
data/raw/transactions.csv   ← PaySim dataset
data/raw/customers.csv      ← Customer behaviour dataset
data/raw/creditcard.csv     ← Credit card PCA dataset
```

### Step 3 — Run preprocessing
```bash
python ml_pipeline/preprocess.py
```
Outputs `data/processed/transactions_clean.csv` and `creditcard_clean.csv`.

### Step 4 — Train models
```bash
python ml_pipeline/train.py
```
Saves models to `models/` and results to `reports/`.

### Step 5 — Evaluate models
```bash
python ml_pipeline/evaluate.py
```
Saves confusion matrix and ROC curve plots to `reports/`.

### Step 6 — Start the API
```bash
uvicorn backend.app.main:app --reload
```
Visit **http://127.0.0.1:8000/docs** for interactive API docs.

### Step 7 — Open the dashboard
Open `frontend/index.html` in your browser.

### Step 8 — Run tests (optional)
```bash
python tests/test_api.py
```

---

## 🔌 API Endpoints

| Method | Endpoint                          | Description                        |
|--------|-----------------------------------|------------------------------------|
| POST   | `/api/fraud/predict/creditcard`   | Predict fraud on CC transaction    |
| POST   | `/api/fraud/predict/transaction`  | Predict fraud on PaySim transaction|
| GET    | `/api/fraud/stats`                | Fraud statistics summary           |
| GET    | `/api/fraud/history`              | Recent prediction history          |
| GET    | `/api/fraud/alerts`               | Open fraud alerts                  |
| PATCH  | `/api/fraud/alerts/{id}/resolve`  | Resolve an alert                   |

---

## 🧠 ML Pipeline

1. **Preprocessing** — clean, encode, engineer balance-error features, enrich with customer stats
2. **SMOTE** — oversample minority (fraud) class to handle extreme imbalance
3. **Models compared** — Logistic Regression, Random Forest, XGBoost
4. **Best model selected** — XGBoost (highest F1/Recall)
5. **Evaluation** — Precision, Recall, F1, ROC-AUC, Confusion Matrix

---

## 🛠 Technologies

Python · Pandas · NumPy · Scikit-learn · XGBoost · imbalanced-learn  
FastAPI · SQLAlchemy · SQLite/PostgreSQL · Joblib  
HTML · CSS · JavaScript (Vanilla)
