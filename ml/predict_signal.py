from __future__ import annotations
import argparse
import json
import joblib
import pandas as pd

import sys
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from signals.rules import rsi_trend_atr_signal


def main():
    ap = argparse.ArgumentParser(description="Emit latest ML-guided signal JSON from a features CSV.")
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--tp_mult", type=float, default=1.5)
    ap.add_argument("--sl_mult", type=float, default=1.0)
    args = ap.parse_args()

    df = pd.read_csv(args.features_csv, parse_dates=["timestamp"]).dropna().reset_index(drop=True)
    payload = joblib.load(args.model_path)
    model = payload["model"]
    cols = payload["features"]
    thr = float(payload["threshold"])

    X = df[cols].astype(float).values
    p = model.predict_proba(X[-1:])[:, 1][0]

    sig = rsi_trend_atr_signal(df, tp_mult=args.tp_mult, sl_mult=args.sl_mult)
    latest = sig.iloc[-1]
    side = int(1 if p >= thr else 0)
    out = {
        "timestamp": str(df["timestamp"].iloc[-1]),
        "prob_up": float(p),
        "threshold": thr,
        "side": side,
        "tp": float(latest["tp"]) if side == 1 else None,
        "sl": float(latest["sl"]) if side == 1 else None,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

