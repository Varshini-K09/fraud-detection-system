"""
models.py
---------
SQLAlchemy ORM models for transactions, predictions, and alerts.
"""

from sqlalchemy import Column, Integer, Float, String, Boolean, DateTime, Text
from sqlalchemy.sql import func
from .database import Base


class Transaction(Base):
    __tablename__ = "transactions"

    id             = Column(Integer, primary_key=True, index=True)
    dataset        = Column(String(20), default="cc")        # 'cc' or 'txn'
    amount         = Column(Float)
    transaction_type = Column(String(20), nullable=True)
    step           = Column(Integer, nullable=True)
    raw_features   = Column(Text, nullable=True)              # JSON string
    created_at     = Column(DateTime(timezone=True), server_default=func.now())


class Prediction(Base):
    __tablename__ = "predictions"

    id                 = Column(Integer, primary_key=True, index=True)
    transaction_id     = Column(Integer, index=True)
    fraud_probability  = Column(Float)
    is_fraud           = Column(Boolean)
    risk_level         = Column(String(10))
    model_used         = Column(String(30), default="XGBoost")
    created_at         = Column(DateTime(timezone=True), server_default=func.now())


class Alert(Base):
    __tablename__ = "alerts"

    id             = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(Integer, index=True)
    risk_level     = Column(String(10))
    message        = Column(Text)
    resolved       = Column(Boolean, default=False)
    created_at     = Column(DateTime(timezone=True), server_default=func.now())
