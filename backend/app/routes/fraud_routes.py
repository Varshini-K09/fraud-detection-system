"""
fraud_routes.py
---------------
All fraud detection API endpoints.
"""

import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from .. import models, schemas
from ..predict import predict_creditcard, predict_transaction

router = APIRouter(prefix="/api/fraud", tags=["Fraud Detection"])


def _save_and_alert(db, txn_id, result):
    """Store prediction and raise alert if HIGH/MEDIUM risk."""
    prediction = models.Prediction(
        transaction_id    = txn_id,
        fraud_probability = result["fraud_probability"],
        is_fraud          = result["is_fraud"],
        risk_level        = result["risk_level"],
    )
    db.add(prediction)

    if result["risk_level"] in ("HIGH", "MEDIUM"):
        alert = models.Alert(
            transaction_id = txn_id,
            risk_level     = result["risk_level"],
            message        = (f"Suspicious transaction detected! "
                              f"Fraud probability: {result['fraud_probability']:.2%}"),
        )
        db.add(alert)

    db.commit()
    db.refresh(prediction)
    return prediction


# ── Credit Card endpoint ───────────────────────────────────────────────────

@router.post("/predict/creditcard", response_model=schemas.PredictionResponse)
def predict_cc(payload: schemas.CreditCardInput, db: Session = Depends(get_db)):
    try:
        result = predict_creditcard(payload.dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    txn = models.Transaction(
        dataset      = "cc",
        amount       = payload.Amount,
        raw_features = json.dumps(payload.dict()),
    )
    db.add(txn)
    db.flush()

    pred = _save_and_alert(db, txn.id, result)

    msg = "🚨 FRAUD ALERT" if result["is_fraud"] else "✅ Transaction appears legitimate"
    return schemas.PredictionResponse(
        transaction_id    = txn.id,
        fraud_probability = result["fraud_probability"],
        is_fraud          = result["is_fraud"],
        risk_level        = result["risk_level"],
        message           = msg,
    )


# ── Transaction (PaySim-style) endpoint ───────────────────────────────────

@router.post("/predict/transaction", response_model=schemas.PredictionResponse)
def predict_txn(payload: schemas.TransactionInput, db: Session = Depends(get_db)):
    try:
        result = predict_transaction(payload.dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    txn = models.Transaction(
        dataset          = "txn",
        amount           = payload.amount,
        transaction_type = payload.transaction_type,
        step             = payload.step,
        raw_features     = json.dumps(payload.dict()),
    )
    db.add(txn)
    db.flush()

    pred = _save_and_alert(db, txn.id, result)

    msg = "🚨 FRAUD ALERT" if result["is_fraud"] else "✅ Transaction appears legitimate"
    return schemas.PredictionResponse(
        transaction_id    = txn.id,
        fraud_probability = result["fraud_probability"],
        is_fraud          = result["is_fraud"],
        risk_level        = result["risk_level"],
        message           = msg,
    )


# ── Analytics endpoints ────────────────────────────────────────────────────

@router.get("/stats", response_model=schemas.StatsResponse)
def get_stats(db: Session = Depends(get_db)):
    total  = db.query(models.Transaction).count()
    frauds = db.query(models.Prediction).filter(models.Prediction.is_fraud == True).count()
    alerts = db.query(models.Alert).filter(models.Alert.resolved == False).count()
    rate   = round(frauds / total, 4) if total > 0 else 0.0
    return schemas.StatsResponse(
        total_transactions = total,
        fraud_count        = frauds,
        fraud_rate         = rate,
        recent_alerts      = alerts,
    )


@router.get("/history")
def get_history(limit: int = 20, db: Session = Depends(get_db)):
    preds = (db.query(models.Prediction, models.Transaction)
               .join(models.Transaction, models.Prediction.transaction_id == models.Transaction.id)
               .order_by(models.Prediction.id.desc())
               .limit(limit)
               .all())
    result = []
    for pred, txn in preds:
        result.append({
            "transaction_id":    txn.id,
            "amount":            txn.amount,
            "dataset":           txn.dataset,
            "fraud_probability": pred.fraud_probability,
            "is_fraud":          pred.is_fraud,
            "risk_level":        pred.risk_level,
            "created_at":        str(pred.created_at),
        })
    return result


@router.get("/alerts")
def get_alerts(resolved: bool = False, db: Session = Depends(get_db)):
    alerts = (db.query(models.Alert)
                .filter(models.Alert.resolved == resolved)
                .order_by(models.Alert.id.desc())
                .limit(50)
                .all())
    return [{"id": a.id, "transaction_id": a.transaction_id,
             "risk_level": a.risk_level, "message": a.message,
             "created_at": str(a.created_at)} for a in alerts]


@router.patch("/alerts/{alert_id}/resolve")
def resolve_alert(alert_id: int, db: Session = Depends(get_db)):
    alert = db.query(models.Alert).filter(models.Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    alert.resolved = True
    db.commit()
    return {"message": f"Alert {alert_id} resolved"}
