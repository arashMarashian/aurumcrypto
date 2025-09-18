from __future__ import annotations
import math
import numpy as np
import pandas as pd

# --- Safe TA fallbacks if 'ta' is not installed ---
try:
    from ta.momentum import RSIIndicator
    from ta.trend import EMAIndicator, MACD
    from ta.volatility import BollingerBands, AverageTrueRange
    _HAS_TA = True
except Exception:
    _HAS_TA = False

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).rolling(window).mean()
    down = (-delta.clip(upper=0)).rolling(window).mean()
    rs = up / (down + 1e-12)
    return 100 - (100 / (1 + rs))

def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    hl = (df["high"] - df["low"]).abs()
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Input: normalized OHLCV df with columns timestamp, open, high, low, close, volume, source, symbol, timeframe."""
    out = df.copy()
    close = out["close"]

    # Momentum & trend
    if _HAS_TA:
        out["rsi_14"] = RSIIndicator(close, 14).rsi()
        out["ema_20"] = EMAIndicator(close, 20).ema_indicator()
        out["ema_50"] = EMAIndicator(close, 50).ema_indicator()
        out["ema_200"] = EMAIndicator(close, 200).ema_indicator()
        macd = MACD(close)
        out["macd"] = macd.macd()
        out["macd_signal"] = macd.macd_signal()
        out["macd_diff"] = macd.macd_diff()
        bb = BollingerBands(close, window=20, window_dev=2)
        out["bb_bbm"] = bb.bollinger_mavg()
        out["bb_bbh"] = bb.bollinger_hband()
        out["bb_bbl"] = bb.bollinger_lband()
        out["bb_pctb"] = (close - out["bb_bbl"]) / (out["bb_bbh"] - out["bb_bbl"]).replace(0, np.nan)
        out["atr_14"] = AverageTrueRange(out["high"], out["low"], close, 14).average_true_range()
    else:
        out["rsi_14"] = _rsi(close, 14)
        out["ema_20"] = _ema(close, 20)
        out["ema_50"] = _ema(close, 50)
        out["ema_200"] = _ema(close, 200)
        out["macd"] = _ema(close, 12) - _ema(close, 26)
        out["macd_signal"] = _ema(out["macd"], 9)
        out["macd_diff"] = out["macd"] - out["macd_signal"]
        mid = close.rolling(20).mean()
        std = close.rolling(20).std()
        out["bb_bbm"] = mid
        out["bb_bbh"] = mid + 2 * std
        out["bb_bbl"] = mid - 2 * std
        out["bb_pctb"] = (close - out["bb_bbl"]) / (out["bb_bbh"] - out["bb_bbl"]).replace(0, np.nan)
        out["atr_14"] = _atr(out, 14)

    # Returns & volatility
    out["ret_1"] = close.pct_change(1)
    for k in (3, 5, 10, 20):
        out[f"ret_{k}"] = close.pct_change(k)
        out[f"roll_std_{k}"] = close.pct_change().rolling(k).std()
    out["range_frac_atr"] = (out["high"] - out["low"]) / (out["atr_14"] + 1e-12)
    out["dist_vwap"] = (
        close
        - (out["close"] * out["volume"]).rolling(50).sum() / (out["volume"].rolling(50).sum() + 1e-9)
    )

    # Regime flags
    out["regime_trend"] = (out["ema_50"] > out["ema_200"]).astype(int) - (out["ema_50"] < out["ema_200"]).astype(int)
    out["above_ema20"] = (close > out["ema_20"]).astype(int)

    return out

def add_label_next_k(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Label = sign of next-k cumulative return (+1/-1/0)."""
    out = df.copy()
    fwd = out["close"].pct_change().shift(-1)  # next bar returns
    fwd_k = fwd.rolling(k).sum()
    out[f"y_next{k}_sign"] = np.sign(fwd_k).fillna(0).astype(int)
    return out

def add_label_atr_barrier(
    df: pd.DataFrame, horizon: int = 15, tp_mult: float = 1.0, sl_mult: float = 1.0
) -> pd.DataFrame:
    """
    Triple-barrier lite: within next `horizon` bars, check if price hits
    +tp_mult*ATR above or -sl_mult*ATR below current close first.
    y_atr = +1 (tp first), -1 (sl first), 0 (neither).
    """
    out = df.copy()
    atr = out["atr_14"].fillna(method="ffill")
    close = out["close"]
    tp = close + tp_mult * atr
    sl = close - sl_mult * atr

    # Iterate efficiently using rolling windows
    y = np.zeros(len(out), dtype=int)
    highs = out["high"].to_numpy()
    lows = out["low"].to_numpy()
    for i in range(len(out) - horizon):
        h_slice = highs[i + 1 : i + 1 + horizon]
        l_slice = lows[i + 1 : i + 1 + horizon]
        tp_hit = (h_slice >= tp.iloc[i]).argmax() if (h_slice >= tp.iloc[i]).any() else 0
        sl_hit = (l_slice <= sl.iloc[i]).argmax() if (l_slice <= sl.iloc[i]).any() else 0
        if tp_hit and sl_hit:
            y[i] = 1 if tp_hit < sl_hit else -1
        elif tp_hit:
            y[i] = 1
        elif sl_hit:
            y[i] = -1
        else:
            y[i] = 0
    out["y_atr"] = y
    return out

def build_features(
    df: pd.DataFrame,
    label_mode: str = "nextk",
    k: int = 3,
    horizon: int = 15,
    tp_mult: float = 1.0,
    sl_mult: float = 1.0,
) -> pd.DataFrame:
    x = add_indicators(df)
    if label_mode == "nextk":
        x = add_label_next_k(x, k=k)
    elif label_mode == "atr":
        x = add_label_atr_barrier(x, horizon=horizon, tp_mult=tp_mult, sl_mult=sl_mult)
    else:
        raise ValueError("label_mode must be 'nextk' or 'atr'")
    # Drop very early NaNs from indicators
    x = x.dropna().reset_index(drop=True)
    return x

