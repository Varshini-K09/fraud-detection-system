"""
test_api.py
-----------
Basic tests for the Fraud Detection API.
Run with: pytest tests/test_api.py -v
(API must be running at localhost:8000)
"""

import pytest
import requests

BASE = "http://127.0.0.1:8000"

CC_LEGIT_SAMPLE = {
    "V1": -1.3598, "V2": -0.0728, "V3": 2.5363, "V4": 1.3782,
    "V5": -0.3383, "V6": 0.4624,  "V7": 0.2396, "V8": 0.0987,
    "V9": 0.3638,  "V10": 0.0908, "V11": -0.5516, "V12": -0.6178,
    "V13": -0.9914, "V14": -0.3112, "V15": 1.4682, "V16": -0.4704,
    "V17": 0.2080, "V18": 0.0258, "V19": 0.4040, "V20": 0.2514,
    "V21": -0.0183, "V22": 0.2778, "V23": -0.1105, "V24": 0.0669,
    "V25": 0.1285, "V26": -0.1891, "V27": 0.1336, "V28": -0.0211,
    "Amount": 149.62, "Time": 0,
}

TXN_FRAUD_SAMPLE = {
    "step": 1,
    "transaction_type": "TRANSFER",
    "amount": 181.0,
    "oldbalanceOrg": 181.0,
    "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0,
}


def test_root():
    r = requests.get(f"{BASE}/")
    assert r.status_code == 200
    assert "Fraud Detection API" in r.json()["message"]


def test_health():
    r = requests.get(f"{BASE}/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_stats():
    r = requests.get(f"{BASE}/api/fraud/stats")
    assert r.status_code == 200
    data = r.json()
    assert "total_transactions" in data
    assert "fraud_rate" in data


def test_predict_creditcard():
    r = requests.post(f"{BASE}/api/fraud/predict/creditcard", json=CC_LEGIT_SAMPLE)
    assert r.status_code == 200
    data = r.json()
    assert "fraud_probability" in data
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert data["risk_level"] in ("SAFE", "LOW", "MEDIUM", "HIGH")
    assert isinstance(data["is_fraud"], bool)


def test_predict_transaction():
    r = requests.post(f"{BASE}/api/fraud/predict/transaction", json=TXN_FRAUD_SAMPLE)
    assert r.status_code == 200
    data = r.json()
    assert "fraud_probability" in data
    assert isinstance(data["is_fraud"], bool)


def test_history():
    r = requests.get(f"{BASE}/api/fraud/history?limit=5")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_alerts():
    r = requests.get(f"{BASE}/api/fraud/alerts")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


if __name__ == "__main__":
    print("Running quick smoke tests...")
    for fn in [test_root, test_health, test_stats,
               test_predict_creditcard, test_predict_transaction,
               test_history, test_alerts]:
        try:
            fn()
            print(f"  ✅ {fn.__name__}")
        except Exception as e:
            print(f"  ❌ {fn.__name__}: {e}")
