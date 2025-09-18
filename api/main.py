from __future__ import annotations
from fastapi import FastAPI, HTTPException, Query, Header
from fastapi.responses import JSONResponse
import pandas as pd
import pathlib
import json

from .model_loader import load_model
from .config import ASSETS, get_api_token

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

