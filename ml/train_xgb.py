from __future__ import annotations
import argparse
import pathlib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

import sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from backtest.core import run_backtest, BTConfig
from signals.rules import rsi_trend_atr_signal
from ml.utils import default_feature_cols, save_json


def make_target(df: pd.DataFrame, label_col: str):
    if label_col not in df.columns:
        raise SystemExit(f"Missing label {label_col}")
    y = (df[label_col] > 0).astype(int).values
    return y


def split_walkforward(n, n_splits=4):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return tscv.split(np.arange(n))


def pick_threshold(y_true, p, side_cost_bps=1.5, price_series=None):
    best_t, best_u = 0.5, -1e9
    for t in np.linspace(0.5, 0.7, 21):
        y_hat = (p >= t).astype(int)
        if y_hat.sum() == 0:
            continue
        hit = ((y_hat == 1) & (y_true == 1)).sum()
        miss = ((y_hat == 1) & (y_true == 0)).sum()
        win_rate = hit / max(hit + miss, 1)
        utility = win_rate - (1 - win_rate) - 0.00015
        if utility > best_u:
            best_u, best_t = utility, t
    return float(best_t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--label_col", default="y_next3_sign")
    ap.add_argument("--n_splits", type=int, default=4)
    ap.add_argument("--model_out", default="models/xgb_nextk.joblib")
    ap.add_argument("--meta_out", default="models/xgb_nextk_meta.json")
    ap.add_argument("--bt_out", default="data/ml_trades.csv")
    ap.add_argument("--fee_bps", type=float, default=1.5)
    ap.add_argument("--max_hold", type=int, default=60)
    args = ap.parse_args()

    import joblib

    df = pd.read_csv(args.features_csv, parse_dates=["timestamp"])
    lbl = "y_next3_sign" if args.label_col == "y_next3_sign" else args.label_col
    df = df.dropna(subset=[lbl]).reset_index(drop=True)

    X_cols = default_feature_cols(df)
    y = make_target(df, lbl)
    X = df[X_cols].astype(float).values

    oof_p = np.zeros(len(df))
    splits = list(split_walkforward(len(df), n_splits=args.n_splits))
    models = []
    for k, (tr, te) in enumerate(splits, 1):
        model = XGBClassifier(
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
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[te])[:, 1]
        oof_p[te] = p
        models.append(model)

    auc = roc_auc_score(y, oof_p) if len(np.unique(y)) > 1 else float("nan")
    thr = pick_threshold(y, oof_p)
    y_hat = (oof_p >= thr).astype(int)
    report = classification_report(y, y_hat, output_dict=True, zero_division=0)

    sig_df = rsi_trend_atr_signal(df, rsi_low=35, rsi_high=65, tp_mult=1.5, sl_mult=1.0)
    sig_df.loc[y_hat == 0, "side"] = 0
    trades_df, summary = run_backtest(df, sig_df, cfg=BTConfig(fee_bps=args.fee_bps, max_hold=args.max_hold))

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
    final.fit(X, y)
    pathlib.Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "features": X_cols, "threshold": thr}, args.model_out)

    trades_df.to_csv(args.bt_out, index=False)
    save_json(
        {
            "features_csv": args.features_csv,
            "n_rows": len(df),
            "n_features": len(X_cols),
            "feature_cols": X_cols,
            "auc_oof": float(auc),
            "threshold": thr,
            "report": report,
            "bt_summary": summary,
            "wf_splits": len(splits),
        },
        args.meta_out,
    )

    print("=== ML TRAIN COMPLETE ===")
    print("[auc_oof]", auc)
    print("[threshold]", thr)
    print(
        "[report]",
        {"precision": report["1"]["precision"], "recall": report["1"]["recall"], "f1": report["1"]["f1-score"]},
    )
    print("[bt_summary]", summary)
    print("[written]", args.model_out, args.meta_out, args.bt_out)


if __name__ == "__main__":
    main()

