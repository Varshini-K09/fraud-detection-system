"""
main.py
-------
FastAPI application entry point.
Run with: uvicorn backend.app.main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import engine, Base
from .routes.fraud_routes import router as fraud_router

# Create all DB tables on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="ML-powered fraud detection for financial transactions.",
    version="1.0.0",
)

# Allow frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(fraud_router)


@app.get("/")
def root():
    return {
        "message": "Fraud Detection API is running",
        "docs":    "/docs",
        "endpoints": {
            "predict_creditcard":  "/api/fraud/predict/creditcard",
            "predict_transaction": "/api/fraud/predict/transaction",
            "stats":               "/api/fraud/stats",
            "history":             "/api/fraud/history",
            "alerts":              "/api/fraud/alerts",
        }
    }


@app.get("/health")
def health():
    return {"status": "ok"}
