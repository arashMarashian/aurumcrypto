#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Use venv if present
[ -d .venv ] && source .venv/bin/activate || true
[ -d venv ]  && source venv/bin/activate  || true

echo "[refresh] fetching data..."
python tools/ingest.py gold   --ticker XAUUSD=X --interval 5m --period 30d --out data/xauusd_5m
python tools/ingest.py crypto --symbol BTC/USDT --exchange binance --timeframe 1m --limit 1500 --out data/btcusdt_1m

echo "[refresh] building ATR features..."
python tools/make_features.py --in_csv data/xauusd_5m.csv  --label_mode atr --horizon 15 --tp_mult 1.0 --sl_mult 1.0 --out_base data/xauusd_5m_atr
python tools/make_features.py --in_csv data/btcusdt_1m.csv --label_mode atr --horizon 15 --tp_mult 1.0 --sl_mult 1.0 --out_base data/btcusdt_1m_atr

echo "[refresh] done."
