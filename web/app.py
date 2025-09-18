import streamlit as st
import pandas as pd
import json
import pathlib
from datetime import datetime

st.set_page_config(page_title="Signal Dashboard", layout="wide")

st.title("Gold & BTC Signals")

features_csv = st.text_input("Features CSV path", "data/xauusd_5m_atr.csv")
model_path = st.text_input("Model path", "models/xgb_barrier.joblib")
meta_path = st.text_input("Meta JSON path", "models/xgb_barrier_meta.json")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Latest Signal")
    try:
        import joblib

        df = pd.read_csv(features_csv, parse_dates=["timestamp"]).dropna().reset_index(drop=True)
        if df.empty:
            raise ValueError("Features CSV has no rows.")
        bundle = joblib.load(model_path)
        cols = bundle["features"]
        X = df[cols].astype(float).values
        long_m = bundle["long_model"]
        short_m = bundle["short_model"]
        thr_l = float(bundle["best_thresholds"]["long"])
        thr_s = float(bundle["best_thresholds"]["short"])
        p_long = float(long_m.predict_proba(X[-1:])[:, 1][0])
        p_short = float(short_m.predict_proba(X[-1:])[:, 1][0])
        side = 1 if (p_long >= thr_l and p_long >= p_short) else (-1 if (p_short >= thr_s and p_short > p_long) else 0)
        row = df.iloc[-1]
        close = float(row["close"])
        atr = float(row.get("atr_14", 0.0))
        tp = close + 1.5 * atr if side == 1 else (close - 1.5 * atr if side == -1 else None)
        sl = close - 1.0 * atr if side == 1 else (close + 1.0 * atr if side == -1 else None)

        st.metric("Timestamp", str(row["timestamp"]))
        st.metric("Close", f"{close:,.2f}")
        st.metric("Prob LONG", f"{p_long:.3f}")
        st.metric("Prob SHORT", f"{p_short:.3f}")
        st.metric("Decision", {1: "BUY", -1: "SELL", 0: "FLAT"}[side])
        st.write({"tp": tp, "sl": sl})
    except Exception as e:
        st.error(f"Error: {e}")

with col2:
    st.subheader("Recent Equity (ML backtest)")
    try:
        ml_trades = st.text_input("ML backtest trades CSV (optional)", "data/ml_trades.csv")
        path = pathlib.Path(ml_trades)
        if path.exists():
            tdf = pd.read_csv(path)
            if "net" in tdf.columns:
                tdf["cum_net"] = tdf["net"].cumsum()
                st.line_chart(tdf[["cum_net"]])
            else:
                st.info("Trades CSV missing 'net' column.")
        else:
            st.info("No ml_trades.csv yet. Train first to populate.")
    except Exception as e:
        st.error(f"Plot error: {e}")

st.caption("Tip: refresh features regularly, re-train periodically, and keep thresholds under walk-forward review.")

