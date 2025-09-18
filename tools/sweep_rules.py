from __future__ import annotations
import argparse
import pandas as pd
import numpy as np
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from signals.rules import rsi_trend_atr_signal
from backtest.core import run_backtest, BTConfig
from backtest.metrics import trade_metrics


def walk_forward_blocks(df: pd.DataFrame, n_blocks: int = 4):
    n = len(df)
    block = n // n_blocks
    for i in range(n_blocks):
        start = i * block
        end = (i + 1) * block if i < n_blocks - 1 else n
        train = df.iloc[: end - block] if i > 0 else df.iloc[:end]
        test = df.iloc[end - block : end]
        if len(test) < 50:
            continue
        yield i + 1, train, test


def run_once(df: pd.DataFrame, params: dict, cfg: BTConfig):
    sig = rsi_trend_atr_signal(
        df,
        rsi_low=params["rsi_low"],
        rsi_high=params["rsi_high"],
        tp_mult=params["tp_mult"],
        sl_mult=params["sl_mult"],
    )
    trades, _ = run_backtest(df, sig, cfg)
    return trade_metrics(trades)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--rsi_low", default="30,35,40")
    ap.add_argument("--rsi_high", default="60,65,70")
    ap.add_argument("--tp_mult", default="1.0,1.5,2.0")
    ap.add_argument("--sl_mult", default="1.0,1.5")
    ap.add_argument("--fee_bps", type=float, default=1.5)
    ap.add_argument("--max_hold", type=int, default=60)
    ap.add_argument("--session", default=None)
    ap.add_argument("--weekdays", default=None)
    ap.add_argument("--n_blocks", type=int, default=4)
    ap.add_argument("--out_csv", default="data/sweep_rules.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.features_csv, parse_dates=["timestamp"])
    if args.session or args.weekdays:
        from tools.run_rules import _filter_session

        df = _filter_session(df, session=args.session, weekdays=args.weekdays)

    grid = []
    for rl in map(int, args.rsi_low.split(",")):
        for rh in map(int, args.rsi_high.split(",")):
            if rl >= rh:
                continue
            for tp in map(float, args.tp_mult.split(",")):
                for sl in map(float, args.sl_mult.split(",")):
                    grid.append({"rsi_low": rl, "rsi_high": rh, "tp_mult": tp, "sl_mult": sl})

    cfg = BTConfig(fee_bps=args.fee_bps, max_hold=args.max_hold)

    rows = []
    for params in grid:
        full = run_once(df, params, cfg)
        wf_stats = []
        for _, _, test in walk_forward_blocks(df, args.n_blocks):
            wf_stats.append(run_once(test, params, cfg))
        if wf_stats:
            wf = {k: float(np.mean([s[k] for s in wf_stats])) for k in wf_stats[0].keys()}
        else:
            wf = {k: np.nan for k in full.keys()}
        rows.append(
            {
                **params,
                **{f"full_{k}": v for k, v in full.items()},
                **{f"wf_{k}": v for k, v in wf.items()},
            }
        )

    out = pd.DataFrame(rows).sort_values("wf_sharpe", ascending=False)
    out.to_csv(args.out_csv, index=False)
    print(f"[sweep] wrote {args.out_csv} rows={len(out)}")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
