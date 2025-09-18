from __future__ import annotations
import argparse
import pathlib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import joblib

import sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ml.utils import default_feature_cols, save_json
from backtest.core import run_backtest, BTConfig
from signals.rules import rsi_trend_atr_signal


def time_splits(n, n_splits=4):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return list(tscv.split(np.arange(n)))


def prob_head(df: pd.DataFrame, label: str, pos_value: int):
    y_full = df[label].values
    y = (y_full == pos_value).astype(int)
    return y


def train_head(X: np.ndarray, y: np.ndarray, splits, calibrate: bool = True):
    oof = np.zeros(len(y))
    models = []
    for tr, te in splits:
        base = XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
        )
        if calibrate:
            model = CalibratedClassifierCV(base, method="isotonic", cv=3)
        else:
            model = base
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[te])[:, 1]
        oof[te] = p
        models.append(model)
    auc = roc_auc_score(y, oof) if len(np.unique(y)) > 1 else float("nan")
    final = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )
    if calibrate:
        final = CalibratedClassifierCV(final, method="isotonic", cv=3)
    final.fit(X, y)
    return {"oof": oof, "auc": float(auc), "final": final}


def sweep_thresholds(df: pd.DataFrame, p_long: np.ndarray, p_short: np.ndarray, fee_bps: float = 1.5, max_hold: int = 60):
    best = None
    thrs = np.linspace(0.55, 0.8, 11)
    for tl in thrs:
        for ts in thrs:
            sig = pd.DataFrame(index=df.index)
            sig["side"] = 0
            sig.loc[p_long >= tl, "side"] = 1
            sig.loc[p_short >= ts, "side"] = -1

            scaffold = rsi_trend_atr_signal(df)
            sig["tp"] = scaffold["tp"]
            sig["sl"] = scaffold["sl"]

            trades, summary = run_backtest(df, sig, cfg=BTConfig(fee_bps=fee_bps, max_hold=max_hold))
            sharpe = summary.get("sharpe", float("nan"))
            if best is None or sharpe > best["summary"].get("sharpe", float("-inf")):
                best = {"tl": float(tl), "ts": float(ts), "summary": summary}
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--n_splits", type=int, default=4)
    ap.add_argument("--fee_bps", type=float, default=1.5)
    ap.add_argument("--max_hold", type=int, default=60)
    ap.add_argument("--model_out", default="models/xgb_barrier.joblib")
    ap.add_argument("--meta_out", default="models/xgb_barrier_meta.json")
    args = ap.parse_args()

    df = pd.read_csv(args.features_csv, parse_dates=["timestamp"]).dropna().reset_index(drop=True)
    if "y_atr" not in df.columns:
        raise SystemExit("y_atr label missing. Re-run features with --label_mode atr.")

    X_cols = default_feature_cols(df)
    X = df[X_cols].astype(float).values
    splits = time_splits(len(df), args.n_splits)

    y_long = prob_head(df, "y_atr", 1)
    y_short = prob_head(df, "y_atr", -1)

    long_res = train_head(X, y_long, splits, calibrate=True)
    short_res = train_head(X, y_short, splits, calibrate=True)

    best = sweep_thresholds(df, long_res["oof"], short_res["oof"], fee_bps=args.fee_bps, max_hold=args.max_hold)

    payload = {
        "long_model": long_res["final"],
        "short_model": short_res["final"],
        "features": X_cols,
        "best_thresholds": {"long": best["tl"], "short": best["ts"]},
    }
    pathlib.Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, args.model_out)

    meta = {
        "rows": len(df),
        "features": X_cols,
        "auc_oof": {"long": long_res["auc"], "short": short_res["auc"]},
        "best_thresholds": {"long": best["tl"], "short": best["ts"]},
        "bt_summary": best["summary"],
    }
    save_json(meta, args.meta_out)

    print("=== ML BARRIER TRAIN COMPLETE ===")
    print("[auc_oof]", meta["auc_oof"])
    print("[thresholds]", meta["best_thresholds"])
    print("[bt_summary]", meta["bt_summary"])
    print("[written]", args.model_out, args.meta_out)


if __name__ == "__main__":
    main()

