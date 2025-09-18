from __future__ import annotations
from fastapi import FastAPI, HTTPException, Query, Header
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import pathlib
import json

from ta.trend import EMAIndicator
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator

from .model_loader import load_model
from .config import ASSETS, get_api_token
from tools.ingest import fetch_gold, fetch_crypto

import csv, time

LOG_PATH = pathlib.Path("data/signal_log.csv")

def _log_signal(asset, payload):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    first = not LOG_PATH.exists()
    with LOG_PATH.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ts","asset","side","p_long","p_short","thr_long","thr_short","tp","sl"])
        if first: w.writeheader()
        w.writerow({"ts": time.time(), "asset": asset, **{k: payload.get(k) for k in ["side","p_long","p_short","thr_long","thr_short","tp","sl"]}})


app = FastAPI(title="Gold-BTC Signal API", version="0.2.0")


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def index():
    return {
        "service": "Gold-BTC Signal API",
        "assets": sorted(ASSETS.keys()),
        "endpoints": {
            "health": "/health",
            "signal": "/signal?asset=XAU",
            "meta": "/meta?asset=XAU",
            "docs": "/docs",
        },
    }



def _auth(x_token: str | None):
    expected = get_api_token()
    provided = (x_token or "").strip()
    if expected and provided != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing token")


def _latest_row_signal(features_csv: str, model_path: str):
    df = pd.read_csv(features_csv, parse_dates=["timestamp"]).dropna().reset_index(drop=True)
    if df.empty:
        raise HTTPException(status_code=400, detail="No rows in features CSV.")
    bundle = load_model(model_path)
    cols = bundle["features"]
    X = df[cols].astype(float).values
    long_m = bundle["long_model"]
    short_m = bundle["short_model"]
    thr_l = float(bundle["best_thresholds"]["long"])
    thr_s = float(bundle["best_thresholds"]["short"])

    p_long = float(long_m.predict_proba(X[-1:])[:, 1][0])
    p_short = float(short_m.predict_proba(X[-1:])[:, 1][0])

    side = 0
    if p_long >= thr_l and p_long >= p_short:
        side = 1
    elif p_short >= thr_s and p_short > p_long:
        side = -1

    row = df.iloc[-1]
    tp = None
    sl = None
    if "atr_14" in df.columns:
        atr = float(row["atr_14"])
        close = float(row["close"])
        if side == 1:
            tp = close + (1.5 * atr)
            sl = close - (1.0 * atr)
        elif side == -1:
            tp = close - (1.5 * atr)
            sl = close + (1.0 * atr)

    return {
        "timestamp": str(row["timestamp"]),
        "p_long": p_long,
        "p_short": p_short,
        "thr_long": thr_l,
        "thr_short": thr_s,
        "side": side,
        "tp": tp,
        "sl": sl,
    }


@app.get("/signal")
def signal(
    features_csv: str | None = None,
    model_path: str | None = None,
    asset: str | None = Query(None, description="Shortcut: XAU or BTC"),
    x_token: str | None = Header(None),
):
    _auth(x_token)
    if asset:
        a = ASSETS.get(asset.upper())
        if not a:
            raise HTTPException(status_code=404, detail="Unknown asset")
        features_csv = a["features"]
        model_path = a["model"]
    if not features_csv or not model_path:
        raise HTTPException(status_code=400, detail="Provide asset or features_csv+model_path")
    try:
        out = _latest_row_signal(features_csv, model_path)
        _log_signal(asset or features_csv, out)
        return JSONResponse(out)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/meta")
def meta(
    meta_path: str | None = None,
    asset: str | None = Query(None, description="Shortcut: XAU or BTC"),
    x_token: str | None = Header(None),
):
    _auth(x_token)
    if asset:
        a = ASSETS.get(asset.upper())
        if not a:
            raise HTTPException(status_code=404, detail="Unknown asset")
        meta_path = a["meta"]
    if not meta_path:
        raise HTTPException(status_code=400, detail="Provide asset or meta_path")
    p = pathlib.Path(meta_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Meta file not found.")
    return JSONResponse(json.loads(p.read_text()))


@app.get("/bars")
def bars(
    asset: str = Query(..., description="XAU or BTC"),
    n: int = Query(300, ge=50, le=2000, description="number of rows"),
    x_token: str | None = Header(None, alias="X-Token"),
):
    _auth(x_token)
    asset_up = asset.upper()
    if asset_up not in ASSETS:
        raise HTTPException(status_code=404, detail="Unknown asset")

    try:
        if asset_up == "XAU":
            df = fetch_gold(ticker="GC=F", interval="5m", period="5d")
        else:
            df = fetch_crypto(symbol="BTC/USDT", exchange="binance", timeframe="1m", limit=max(n, 600))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"fetch error: {exc}")

    if df is None or df.empty:
        raise HTTPException(status_code=500, detail="no bars")

    df = df.tail(n).copy()
    df["close"] = df["close"].astype(float)

    try:
        df["ema_20"] = EMAIndicator(df["close"], window=20).ema_indicator()
        df["ema_50"] = EMAIndicator(df["close"], window=50).ema_indicator()
        bb = BollingerBands(df["close"], window=20, window_dev=2)
        df["bb_mid"] = bb.bollinger_mavg()
        df["bb_up"] = bb.bollinger_hband()
        df["bb_low"] = bb.bollinger_lband()
        df["rsi_14"] = RSIIndicator(df["close"], window=14).rsi()
    except Exception:
        pass

    def _series(name: str):
        if name in df.columns:
            return df[name].astype(float).round(8).replace([np.inf, -np.inf], np.nan).fillna(np.nan).tolist()
        return [np.nan] * len(df)

    out = {
        "symbol": str(df.get("symbol", pd.Series([asset_up])).iloc[-1]),
        "source": str(df.get("source", pd.Series(["unknown"])).iloc[-1]),
        "rows": len(df),
        "timestamp": pd.to_datetime(df["timestamp"]).astype(str).tolist(),
        "open": df["open"].astype(float).round(8).fillna(0.0).tolist(),
        "high": df["high"].astype(float).round(8).fillna(0.0).tolist(),
        "low": df["low"].astype(float).round(8).fillna(0.0).tolist(),
        "close": df["close"].astype(float).round(8).fillna(0.0).tolist(),
        "ema_20": [x if np.isfinite(x) else None for x in _series("ema_20")],
        "ema_50": [x if np.isfinite(x) else None for x in _series("ema_50")],
        "bb_mid": [x if np.isfinite(x) else None for x in _series("bb_mid")],
        "bb_up": [x if np.isfinite(x) else None for x in _series("bb_up")],
        "bb_low": [x if np.isfinite(x) else None for x in _series("bb_low")],
        "rsi_14": [x if np.isfinite(x) else None for x in _series("rsi_14")],
    }
    return JSONResponse(out)
