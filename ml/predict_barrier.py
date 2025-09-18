from __future__ import annotations
import argparse
import json
import joblib
import pandas as pd
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def main():
    ap = argparse.ArgumentParser(description="Emit latest ML long/short signal using barrier heads.")
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--model_path", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.features_csv, parse_dates=["timestamp"]).dropna().reset_index(drop=True)
    bundle = joblib.load(args.model_path)
    cols = bundle["features"]
    long_m = bundle["long_model"]
    short_m = bundle["short_model"]
    thr_l = float(bundle["best_thresholds"]["long"])
    thr_s = float(bundle["best_thresholds"]["short"])

    X = df[cols].astype(float).values
    p_long = float(long_m.predict_proba(X[-1:])[:, 1][0])
    p_short = float(short_m.predict_proba(X[-1:])[:, 1][0])

    side = 0
    if p_long >= thr_l and p_long >= p_short:
        side = 1
    elif p_short >= thr_s and p_short > p_long:
        side = -1

    out = {
        "timestamp": str(df["timestamp"].iloc[-1]),
        "p_long": p_long,
        "p_short": p_short,
        "thr_long": thr_l,
        "thr_short": thr_s,
        "side": side,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

