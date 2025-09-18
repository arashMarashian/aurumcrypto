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

## Data Ingestion (Step 2)

Create a venv and install deps:
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Fetch BTC (Binance) 1m candles:

```bash
python tools/ingest.py crypto --symbol BTC/USDT --exchange binance --timeframe 1m --limit 500
```

Fetch Gold via Yahoo (choose one):

```bash
# Spot proxy
python tools/ingest.py gold --ticker XAUUSD=X --interval 5m --period 30d
# or GLD ETF (market hours only)
python tools/ingest.py gold --ticker GLD --interval 5m --period 30d
```

Outputs are saved under /data as CSV and Parquet with normalized schema:
timestamp, open, high, low, close, volume, source, symbol, timeframe

## Features & Labels (Step 3)

Generate indicators and labels:
```bash
# Next-k label (default k=3)
python tools/make_features.py --in_csv data/sample_btcusdt_1m.csv --label_mode nextk --k 3 --out_base data/btcusdt_1m_nextk
python tools/make_features.py --in_csv data/sample_xauusd_5m.csv  --label_mode nextk --k 3 --out_base data/xauusd_5m_nextk

# ATR-barrier label (TP=1*ATR, SL=1*ATR, horizon=15)
python tools/make_features.py --in_csv data/sample_btcusdt_1m.csv --label_mode atr --horizon 15 --tp_mult 1.0 --sl_mult 1.0 --out_base data/btcusdt_1m_atr
```

Outputs include engineered indicators (RSI/EMA/MACD/Bollinger/ATR), returns, regime flags, and a label column (y_next3_sign or y_atr).

## Baseline Rule Signals (Step 4)

Run RSI + trend + ATR SL/TP baseline with backtest:
```bash
python tools/run_rules.py --features_csv data/xauusd_5m_nextk.csv --rsi_low 35 --rsi_high 65 --tp_mult 1.5 --sl_mult 1.0 --fee_bps 1.5 --max_hold 60 --out_trades data/xauusd_rules_trades.csv
```

Outputs: head/tail of trades and a summary dict (trades, hit_rate, avg_net, sharpe, max_dd, cum_net).

## Step 5 – Backtesting upgrades & tuning

Session filter + vol targeting:
```bash
python tools/run_rules.py --features_csv data/xauusd_5m_nextk.csv --session "07:00-20:00" --weekdays "1-5" --vol_target 0.10 --out_trades data/xauusd_rules_trades.csv
```

This applies a session filter (e.g., gold market hours), optional volatility targeting, saves trades and an equity/drawdown plot PNG.

Grid search + walk-forward:

```bash
python tools/sweep_rules.py --features_csv data/xauusd_5m_nextk.csv --session "07:00-20:00" --weekdays "1-5" --out_csv data/sweep_rules.csv
```

Outputs a CSV ranking parameter sets by full-sample and walk-forward metrics.

## Step 6 – ML Baseline (XGBoost)

Train with walk-forward OOF, auto thresholding, and sanity backtest:
```bash
python ml/train_xgb.py --features_csv data/xauusd_5m_nextk.csv --model_out models/xgb_nextk.joblib --meta_out models/xgb_nextk_meta.json --bt_out data/ml_trades.csv
```

Emit the latest ML-guided signal as JSON:

```bash
python ml/predict_signal.py --features_csv data/xauusd_5m_nextk.csv --model_path models/xgb_nextk.joblib
```

## Step 6b – ML with triple-barrier (long & short heads)

Train calibrated XGBoost heads (long/short) on `y_atr` and sweep thresholds:
```bash
# Recompute features if needed to include y_atr:
python tools/make_features.py --in_csv data/sample_xauusd_5m.csv --label_mode atr --horizon 15 --tp_mult 1.0 --sl_mult 1.0 --out_base data/xauusd_5m_atr

python ml/train_xgb_barrier.py --features_csv data/xauusd_5m_atr.csv --model_out models/xgb_barrier.joblib --meta_out models/xgb_barrier_meta.json
```

Emit latest signal (long/short/flat):

```bash
python ml/predict_barrier.py --features_csv data/xauusd_5m_atr.csv --model_path models/xgb_barrier.joblib
```

## Step 7 – API & Dashboard

Run API:
```bash
uvicorn api.main:app --reload --port 8000
# test:
curl "http://127.0.0.1:8000/signal?features_csv=data/xauusd_5m_atr.csv&model_path=models/xgb_barrier.joblib"
```

Run dashboard:

```bash
streamlit run web/app.py
```

## Step 8 – Multi-asset + API token + refresh

1) Copy `.env.example` → `.env` and set `API_TOKEN`.
2) Start API:
   ```bash
   export $(grep -v '^#' .env | xargs)
   uvicorn api.main:app --reload --port 8000
   ```
3) Refresh data & features:
   ```bash
   bash tools/refresh_assets.sh
   ```
4) Call API with token:
   ```bash
   curl -s -H "X-Token: $API_TOKEN" "http://127.0.0.1:8000/signal?asset=XAU"
   curl -s -H "X-Token: $API_TOKEN" "http://127.0.0.1:8000/signal?asset=BTC"
   ```

Verify Step 8
```bash
# 1) set token in env
cp .env.example .env && sed -i 's/changeme/mytoken123/' .env
export $(grep -v '^#' .env | xargs)

# 2) refresh data/features (uses public endpoints)
bash tools/refresh_assets.sh

# 3) run API
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

# 4) test endpoints (with token)
curl -s -H "X-Token: $API_TOKEN" "http://127.0.0.1:8000/health"
curl -s -H "X-Token: $API_TOKEN" "http://127.0.0.1:8000/signal?asset=XAU" | python -m json.tool
curl -s -H "X-Token: $API_TOKEN" "http://127.0.0.1:8000/signal?asset=BTC" | python -m json.tool
curl -s -H "X-Token: $API_TOKEN" "http://127.0.0.1:8000/meta?asset=XAU"   | python -m json.tool
```

