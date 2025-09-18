from __future__ import annotations
import numpy as np
import pandas as pd


def rsi_trend_atr_signal(
    df: pd.DataFrame,
    rsi_low: int = 35,
    rsi_high: int = 65,
    tp_mult: float = 1.5,
    sl_mult: float = 1.0,
):
    """
    Expect columns: timestamp, close, rsi_14, ema_50, ema_200, atr_14
    Returns a DataFrame with columns: 'side' (+1/-1/0), 'tp', 'sl'
    Logic:
      - Long if rsi<rsi_low AND ema50>ema200
      - Short if rsi>rsi_high AND ema50<ema200
      - Else flat
      - TP/SL = close +/- multipliers * ATR in direction of trade
    """
    out = pd.DataFrame(index=df.index)
    trend = np.where(df["ema_50"] > df["ema_200"], 1, np.where(df["ema_50"] < df["ema_200"], -1, 0))
    long_cond = (df["rsi_14"] < rsi_low) & (trend > 0)
    short_cond = (df["rsi_14"] > rsi_high) & (trend < 0)
    side = np.where(long_cond, 1, np.where(short_cond, -1, 0))
    out["side"] = side

    close = df["close"].to_numpy()
    atr = df["atr_14"].fillna(method="ffill").to_numpy()
    tp = np.where(
        side == 1,
        close + tp_mult * atr,
        np.where(side == -1, close - tp_mult * atr, np.nan),
    )
    sl = np.where(
        side == 1,
        close - sl_mult * atr,
        np.where(side == -1, close + sl_mult * atr, np.nan),
    )
    out["tp"] = tp
    out["sl"] = sl
    return out

