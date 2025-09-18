# aurumcrypto
Python-based toolkit for short-term trading signals on Bitcoin (BTC) and Gold (XAUUSD). Includes data ingestion, technical indicators, ML-based models, and rule-based strategies. Features backtesting with costs/slippage, FastAPI signal API, and a web dashboard for real-time signal monitoring.

# Gold-BTC Signal Lab

Python-based toolkit for short-term trading signals on Bitcoin (BTC) and Gold (XAUUSD).

## Features
- Data ingestion (ccxt/yfinance)
- Technical & ML feature engineering
- Rule-based + ML signals (buy/sell/flat)
- Backtesting with costs/slippage
- FastAPI signal API
- Web dashboard (Streamlit/Next.js)

## Structure
- /data        raw & processed (ignored)
- /signals     signal generation
- /backtest    backtesting engine
- /api         FastAPI app
- /web         dashboard
- /models      trained models (ignored)
- /notebooks   research & EDA
- /tools       utilities

## Quickstart (dev)
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Roadmap

Setup & structure ✅

Data ingestion

Features & labels

Baseline rules

Backtesting

ML models

API

Web UI

