from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt


def plot_equity_drawdown(trades: pd.DataFrame, out_png: str):
    if trades.empty:
        print("[plot] no trades, skipping plot")
        return
    eq = trades["net"].cumsum()
    roll = eq.cummax()
    dd = eq - roll

    plt.figure()
    eq.plot(label="Equity")
    dd.plot(label="Drawdown")
    plt.legend()
    plt.xlabel("Trade #")
    plt.ylabel("PnL (points)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()
    print(f"[plot] saved {out_png}")

