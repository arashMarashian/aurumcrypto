from __future__ import annotations
import os

API_TOKEN = os.getenv("API_TOKEN", "")  # set in .env

# Default asset routing (adjust paths as you like)
ASSETS = {
    "XAU": {
        "features": "data/xauusd_5m_atr.csv",
        "model": "models/xgb_barrier.joblib",
        "meta": "models/xgb_barrier_meta.json",
    },
    "BTC": {
        "features": "data/btcusdt_1m_atr.csv",
        "model": "models/xgb_barrier.joblib",  # reuse until you train a BTC model
        "meta": "models/xgb_barrier_meta.json",
    },
}
