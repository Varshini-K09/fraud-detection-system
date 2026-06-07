-- schema.sql
-- PostgreSQL schema for production deployment
-- (SQLite is used locally via SQLAlchemy)

CREATE TABLE IF NOT EXISTS transactions (
    id               SERIAL PRIMARY KEY,
    dataset          VARCHAR(20) DEFAULT 'cc',
    amount           FLOAT,
    transaction_type VARCHAR(20),
    step             INTEGER,
    raw_features     TEXT,
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS predictions (
    id                SERIAL PRIMARY KEY,
    transaction_id    INTEGER REFERENCES transactions(id),
    fraud_probability FLOAT,
    is_fraud          BOOLEAN,
    risk_level        VARCHAR(10),
    model_used        VARCHAR(30) DEFAULT 'XGBoost',
    created_at        TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS alerts (
    id             SERIAL PRIMARY KEY,
    transaction_id INTEGER REFERENCES transactions(id),
    risk_level     VARCHAR(10),
    message        TEXT,
    resolved       BOOLEAN DEFAULT FALSE,
    created_at     TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_predictions_transaction ON predictions(transaction_id);
CREATE INDEX idx_predictions_fraud       ON predictions(is_fraud);
CREATE INDEX idx_alerts_resolved         ON alerts(resolved);
