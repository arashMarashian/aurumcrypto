from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class BTConfig:
    fee_bps: float = 1.5   # round-trip cost approximation in basis points
    max_hold: int = 60     # bars; safety cap
    risk_per_trade: float = 1.0  # notional units (PnL measured in price points)


def run_backtest(df: pd.DataFrame, signal_df: pd.DataFrame, cfg: BTConfig = BTConfig()):
    """
    df must contain: timestamp, open, high, low, close
    signal_df must contain: side, tp, sl (aligned by index with df)
    Entry rule: when side != 0 at bar t, enter at OPEN of bar t+1
    Exit rule: first touch of TP/SL using high/low of future bars or timeout at max_hold
    PnL in 'points' on close price units; apply round-trip fees.
    """
    ts = df["timestamp"].reset_index(drop=True)
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    side = signal_df["side"].to_numpy()
    tp = signal_df["tp"].to_numpy()
    sl = signal_df["sl"].to_numpy()

    trades = []
    i = 0
    n = len(df)
    fee = cfg.fee_bps / 10000.0

    while i < n - 2:
        if side[i] == 0:
            i += 1
            continue
        entry_i = i + 1
        entry_px = o[entry_i]
        s = side[i]
        tgt = tp[i]
        stp = sl[i]

        exit_i = min(entry_i + cfg.max_hold, n - 1)
        filled = False
        for j in range(entry_i, exit_i + 1):
            if s == 1:
                if h[j] >= tgt:
                    exit_px = tgt
                    exit_j = j
                    outcome = "tp"
                    filled = True
                    break
                if l[j] <= stp:
                    exit_px = stp
                    exit_j = j
                    outcome = "sl"
                    filled = True
                    break
            else:
                if l[j] <= tgt:
                    exit_px = tgt
                    exit_j = j
                    outcome = "tp"
                    filled = True
                    break
                if h[j] >= stp:
                    exit_px = stp
                    exit_j = j
                    outcome = "sl"
                    filled = True
                    break
        if not filled:
            exit_j = exit_i
            exit_px = c[exit_j]
            outcome = "timeout"

        gross = (exit_px - entry_px) * s
        net = gross - abs(entry_px) * fee
        trades.append(
            {
                "entry_idx": int(entry_i),
                "exit_idx": int(exit_j),
                "entry_time": str(ts.iloc[entry_i]),
                "exit_time": str(ts.iloc[exit_j]),
                "side": int(s),
                "entry": float(entry_px),
                "exit": float(exit_px),
                "outcome": outcome,
                "gross": float(gross),
                "net": float(net),
            }
        )
        i = exit_j + 1

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df, {"trades": 0, "sharpe": 0.0, "hit_rate": 0.0, "avg_net": 0.0, "max_dd": 0.0}

    eq = trades_df["net"].cumsum()
    roll_max = eq.cummax()
    dd = eq - roll_max
    max_dd = float(dd.min())

    mu = trades_df["net"].mean()
    sd = trades_df["net"].std(ddof=1)
    sharpe = float(mu / (sd + 1e-12) * np.sqrt(max(len(trades_df), 1)))

    hit_rate = float((trades_df["net"] > 0).mean())
    summary = {
        "trades": int(len(trades_df)),
        "hit_rate": hit_rate,
        "avg_net": float(mu),
        "sharpe": sharpe,
        "max_dd": max_dd,
        "cum_net": float(trades_df["net"].sum()),
    }
    return trades_df, summary

