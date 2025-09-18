from __future__ import annotations
import os, time, math, pathlib
from typing import List, Dict, Any
import pandas as pd

DATA_DIR = pathlib.Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def ensure_dir(p: pathlib.Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def _normalize_series(obj: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(obj, pd.DataFrame):
        if obj.empty:
            return pd.Series(dtype="float64")
        obj = obj.iloc[:, 0]
    return pd.to_numeric(obj, errors="coerce")


def _numeric_column(df: pd.DataFrame, *names: str) -> pd.Series:
    for name in names:
        if name in df.columns:
            return _normalize_series(df[name])
    if isinstance(df.columns, pd.MultiIndex):
        for name in names:
            for col_key in df.columns:
                if not isinstance(col_key, tuple):
                    continue
                if col_key[-1] == name or col_key[0] == name:
                    return _normalize_series(df[col_key])
                if col_key[-1].lower() == name.lower() or col_key[0].lower() == name.lower():
                    return _normalize_series(df[col_key])
    return pd.Series(index=df.index, dtype="float64")


def normalize_ohlcv_df(df: pd.DataFrame, source: str, symbol: str, timeframe: str) -> pd.DataFrame:
    # Expect columns: timestamp(ms/ns or datetime), open, high, low, close, volume
    # Convert timestamp to UTC ISO datetime
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
    elif "Datetime" in df.columns:
        ts = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
    elif "Date" in df.columns:
        ts = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    else:
        ts = pd.to_datetime(df.index, utc=True, errors="coerce")
    out = pd.DataFrame({
        "timestamp": ts,
        "open": _numeric_column(df, "open", "Open"),
        "high": _numeric_column(df, "high", "High"),
        "low": _numeric_column(df, "low", "Low"),
        "close": _numeric_column(df, "close", "Close"),
        "volume": _numeric_column(df, "volume", "Volume"),
    })
    out["source"] = source
    out["symbol"] = symbol
    out["timeframe"] = timeframe
    out = out.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp")
    out = out.reset_index(drop=True)
    return out


def save_csv_parquet(df: pd.DataFrame, basepath: pathlib.Path):
    ensure_dir(basepath)
    csv_path = basepath.with_suffix(".csv")
    pq_path = basepath.with_suffix(".parquet")
    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(pq_path, index=False)
    except Exception:
        pq_path = None
    return csv_path, pq_path
