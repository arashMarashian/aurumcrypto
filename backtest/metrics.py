from __future__ import annotations
import numpy as np
import pandas as pd


def trade_metrics(trades: pd.DataFrame):
    if trades.empty:
        return {
            "trades": 0,
            "hit_rate": 0.0,
            "avg_net": 0.0,
            "sharpe": 0.0,
            "max_dd": 0.0,
            "cum_net": 0.0,
            "long_trades": 0,
            "short_trades": 0,
            "long_hit": 0.0,
            "short_hit": 0.0,
            "avg_hold_bars": 0.0,
        }
    eq = trades["net"].cumsum()
    roll_max = eq.cummax()
    dd = eq - roll_max
    long_mask = trades["side"] == 1
    short_mask = trades["side"] == -1
    mu = trades["net"].mean()
    sd = trades["net"].std(ddof=1)
    return {
        "trades": int(len(trades)),
        "hit_rate": float((trades["net"] > 0).mean()),
        "avg_net": float(mu),
        "sharpe": float(mu / (sd + 1e-12) * np.sqrt(max(len(trades), 1))),
        "max_dd": float(dd.min()),
        "cum_net": float(trades["net"].sum()),
        "long_trades": int(long_mask.sum()),
        "short_trades": int(short_mask.sum()),
        "long_hit": float((trades.loc[long_mask, "net"] > 0).mean() if long_mask.any() else 0.0),
        "short_hit": float((trades.loc[short_mask, "net"] > 0).mean() if short_mask.any() else 0.0),
        "avg_hold_bars": float((trades["exit_idx"] - trades["entry_idx"]).mean()),
    }

