from __future__ import annotations
import argparse
import pathlib
import sys
import numpy as np
import datetime as dt
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from signals.rules import rsi_trend_atr_signal
from backtest.core import run_backtest, BTConfig
from backtest.metrics import trade_metrics
from backtest.plots import plot_equity_drawdown


def _filter_session(df: pd.DataFrame, session: str | None = None, weekdays: str | None = None) -> pd.DataFrame:
    out = df.copy()
    if session:
        s_from, s_to = session.split("-")
        h1, m1 = map(int, s_from.split(":"))
        h2, m2 = map(int, s_to.split(":"))
        t = out["timestamp"].dt.time
        out = out[(t >= dt.time(h1, m1)) & (t <= dt.time(h2, m2))]
    if weekdays:
        parts = weekdays.split("-")
        if len(parts) == 2:
            lo, hi = map(int, parts)
            lo -= 1
            hi -= 1
            out = out[out["timestamp"].dt.weekday.between(lo, hi)]
    return out


def main():
    ap = argparse.ArgumentParser(description="Run baseline rule-based strategy on features CSV.")
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--rsi_low", type=int, default=35)
    ap.add_argument("--rsi_high", type=int, default=65)
    ap.add_argument("--tp_mult", type=float, default=1.5)
    ap.add_argument("--sl_mult", type=float, default=1.0)
    ap.add_argument("--fee_bps", type=float, default=1.5)
    ap.add_argument("--max_hold", type=int, default=60)
    ap.add_argument("--session", default=None, help='HH:MM-HH:MM, e.g., "07:00-20:00"')
    ap.add_argument("--weekdays", default=None, help='"1-5" for Mon-Fri')
    ap.add_argument("--vol_target", type=float, default=0.0, help="approx per-trade vol target (0=off)")
    ap.add_argument("--out_trades", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.features_csv, parse_dates=["timestamp"])
    needed = ["timestamp", "open", "high", "low", "close", "rsi_14", "ema_50", "ema_200", "atr_14"]
    for col in needed:
        if col not in df.columns:
            raise SystemExit(f"Missing required column: {col}")

    orig_len = len(df)
    df = _filter_session(df, session=args.session, weekdays=args.weekdays)
    if len(df) < 50:
        print("[run_rules] WARNING: too few rows after session filter.")

    sig = rsi_trend_atr_signal(
        df,
        rsi_low=args.rsi_low,
        rsi_high=args.rsi_high,
        tp_mult=args.tp_mult,
        sl_mult=args.sl_mult,
    )
    trades_df, summary = run_backtest(
        df,
        sig,
        cfg=BTConfig(fee_bps=args.fee_bps, max_hold=args.max_hold),
    )

    if args.vol_target and args.vol_target > 0 and not trades_df.empty:
        mult = []
        for _, row in trades_df.iterrows():
            i = int(row["entry_idx"])
            atr = float(df["atr_14"].iloc[i])
            close_px = float(df["close"].iloc[i])
            sigma = atr / max(close_px, 1e-9)
            pos = args.vol_target / max(sigma, 1e-9)
            pos = float(np.clip(pos, 0.25, 2.0))
            mult.append(pos)
        trades_df["net"] = trades_df["net"] * np.array(mult)
        trades_df["gross"] = trades_df["gross"] * np.array(mult)
        trades_df["pos_mult"] = mult

    params = dict(
        rsi_low=args.rsi_low,
        rsi_high=args.rsi_high,
        tp_mult=args.tp_mult,
        sl_mult=args.sl_mult,
    )
    print("[rules] params:", params)
    print("[rules] trades:")
    print(trades_df.head(5).to_string(index=False) if not trades_df.empty else "(no trades)")
    print("...")
    print(trades_df.tail(5).to_string(index=False) if not trades_df.empty else "(no trades)")
    summary = trade_metrics(trades_df)
    print("[summary]", summary)

    if args.out_trades:
        outp = pathlib.Path(args.out_trades)
        outp.parent.mkdir(parents=True, exist_ok=True)
        trades_df.to_csv(outp, index=False)
        print(f"[written] {outp}")
        png_path = str(outp.with_suffix(".png"))
        plot_equity_drawdown(trades_df, png_path)
        print(f"[plot] {png_path}")


if __name__ == "__main__":
    main()
