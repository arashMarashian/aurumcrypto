from __future__ import annotations
import argparse
import pathlib
import sys
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from common import DATA_DIR, save_csv_parquet
from signals.features import build_features


def main():
    ap = argparse.ArgumentParser(description="Generate features from normalized OHLCV CSV.")
    ap.add_argument("--in_csv", required=True, help="Path to normalized OHLCV CSV")
    ap.add_argument("--label_mode", choices=["nextk", "atr"], default="nextk")
    ap.add_argument("--k", type=int, default=3, help="k for nextk label")
    ap.add_argument("--horizon", type=int, default=15, help="horizon for atr label")
    ap.add_argument("--tp_mult", type=float, default=1.0)
    ap.add_argument("--sl_mult", type=float, default=1.0)
    ap.add_argument("--out_base", default=None, help="Output base path without extension")
    args = ap.parse_args()

    src = pathlib.Path(args.in_csv)
    df = pd.read_csv(src)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    feats = build_features(
        df,
        label_mode=args.label_mode,
        k=args.k,
        horizon=args.horizon,
        tp_mult=args.tp_mult,
        sl_mult=args.sl_mult,
    )

    if args.out_base:
        base = pathlib.Path(args.out_base)
    else:
        stem = src.stem.replace(".csv", "")
        base = DATA_DIR / f"{stem}_features"

    csv, pq = save_csv_parquet(feats, base)
    label_cols = [c for c in feats.columns if c.startswith("y_")]
    print(f"[features] rows={len(feats)} cols={len(feats.columns)} labels={label_cols}")
    print("[features] head:")
    print(feats.head(3).to_string(index=False))
    print("...")
    print("[features] tail:")
    print(feats.tail(3).to_string(index=False))
    nulls = feats.isna().sum().sum()
    print(f"[features] total_nulls={nulls}")
    for lc in label_cols:
        vc = feats[lc].value_counts(dropna=False).to_dict()
        print(f"[features] label_dist {lc}: {vc}")
    print(f"[features] written: {csv}{' ' + str(pq) if pq else ''}")


if __name__ == "__main__":
    main()
