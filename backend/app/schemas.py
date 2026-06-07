"""
schemas.py
----------
Pydantic request/response schemas for FastAPI endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime


# ── Credit Card prediction request ────────────────────────────────────────

class CreditCardInput(BaseModel):
    """
    PCA-anonymised credit card transaction.
    V1–V28 are PCA components; Amount and Time are raw.
    """
    V1:  float = 0.0; V2:  float = 0.0; V3:  float = 0.0; V4:  float = 0.0
    V5:  float = 0.0; V6:  float = 0.0; V7:  float = 0.0; V8:  float = 0.0
    V9:  float = 0.0; V10: float = 0.0; V11: float = 0.0; V12: float = 0.0
    V13: float = 0.0; V14: float = 0.0; V15: float = 0.0; V16: float = 0.0
    V17: float = 0.0; V18: float = 0.0; V19: float = 0.0; V20: float = 0.0
    V21: float = 0.0; V22: float = 0.0; V23: float = 0.0; V24: float = 0.0
    V25: float = 0.0; V26: float = 0.0; V27: float = 0.0; V28: float = 0.0
    Amount: float = Field(..., gt=0, description="Transaction amount in USD")
    Time:   float = Field(0.0, description="Seconds elapsed from first transaction")

    class Config:
        json_schema_extra = {
            "example": {
                "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
                "V5": -0.34, "V6": 0.46,  "V7": 0.24, "V8": 0.10,
                "V9": 0.36,  "V10": 0.09, "V11": -0.55, "V12": -0.62,
                "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
                "V17": 0.21, "V18": 0.03,  "V19": 0.40, "V20": 0.25,
                "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
                "V25": 0.13,  "V26": -0.19, "V27": 0.13, "V28": -0.02,
                "Amount": 149.62, "Time": 0,
            }
        }


# ── Transaction (PaySim-style) prediction request ──────────────────────────

class TransactionInput(BaseModel):
    step:           int   = Field(..., description="Hour of simulation")
    transaction_type: str = Field(..., description="PAYMENT | TRANSFER | CASH_OUT | DEBIT | CASH_IN")
    amount:         float = Field(..., gt=0)
    oldbalanceOrg:  float = 0.0
    newbalanceOrig: float = 0.0
    oldbalanceDest: float = 0.0
    newbalanceDest: float = 0.0

    class Config:
        json_schema_extra = {
            "example": {
                "step": 1, "transaction_type": "TRANSFER",
                "amount": 181.0, "oldbalanceOrg": 181.0,
                "newbalanceOrig": 0.0, "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
            }
        }


# ── Response schemas ───────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    transaction_id:    int
    fraud_probability: float
    is_fraud:          bool
    risk_level:        str
    message:           str


class StatsResponse(BaseModel):
    total_transactions: int
    fraud_count:        int
    fraud_rate:         float
    recent_alerts:      int
