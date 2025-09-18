from __future__ import annotations

import os
import time
from typing import Any, Dict, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

# ---------- CONFIG ----------
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
API_TOKEN = os.getenv("API_TOKEN", "")
DEFAULTS: Dict[str, Dict[str, Any]] = {
    "XAU": {
        "features_csv": "data/xauusd_5m_atr.csv",
        "display_mult": 1.0,
        "title": "Gold (XAU / GC=F)",
    },
    "BTC": {
        "features_csv": "data/btcusdt_1m_atr.csv",
        "display_mult": 1.0,
        "title": "Bitcoin (BTCUSDT)",
    },
}


# ---------- HELPERS ----------
def api_get(path: str, params: Dict[str, Any] | None = None) -> Tuple[Dict[str, Any], float]:
    headers = {"X-Token": API_TOKEN} if API_TOKEN else {}
    t0 = time.time()
    resp = requests.get(f"{API_URL}{path}", params=params, headers=headers, timeout=10)
    resp.raise_for_status()
    latency_ms = (time.time() - t0) * 1000.0
    return resp.json(), latency_ms

def api_bars(asset: str, n: int = 300):
    headers = {"X-Token": API_TOKEN} if API_TOKEN else {}
    t0 = time.time()
    resp = requests.get(f"{API_URL}/bars", params={"asset": asset, "n": n}, headers=headers, timeout=10)
    resp.raise_for_status()
    latency = (time.time() - t0) * 1000.0
    payload = resp.json()
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(payload["timestamp"]),
        "open": payload["open"],
        "high": payload["high"],
        "low": payload["low"],
        "close": payload["close"],
        "ema_20": payload.get("ema_20", []),
        "ema_50": payload.get("ema_50", []),
        "bb_mid": payload.get("bb_mid", []),
        "bb_up": payload.get("bb_up", []),
        "bb_low": payload.get("bb_low", []),
        "rsi_14": payload.get("rsi_14", []),
    })
    df["symbol"] = payload.get("symbol", asset)
    df["source"] = payload.get("source", "unknown")
    return df, latency



def load_df(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, parse_dates=["timestamp"]).dropna().reset_index(drop=True)


def last_symbol_source(df: pd.DataFrame) -> Tuple[str, str]:
    sym = str(df["symbol"].iloc[-1]) if "symbol" in df.columns else "UNKNOWN"
    src = str(df["source"].iloc[-1]) if "source" in df.columns else "UNKNOWN"
    return sym, src


def format_timedelta(delta: pd.Timedelta) -> str:
    seconds = int(delta.total_seconds())
    if seconds <= 0:
        return "n/a"
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs and not hours:
        parts.append(f"{secs}s")
    return " ".join(parts) or "n/a"


def seconds_to_next_bar(ts: pd.Timestamp, tf_seconds: int) -> int:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    epoch = int(ts.timestamp())
    remainder = epoch % tf_seconds
    return tf_seconds - remainder if remainder else tf_seconds


def make_dashboard_fig(
    df: pd.DataFrame,
    lookback: int,
    mult: float,
    tp: float | None = None,
    sl: float | None = None,
    title: str = "",
) -> go.Figure:
    tail = df.tail(lookback).copy()
    for col in ("open", "high", "low", "close"):
        if col in tail.columns:
            tail[col] = tail[col] * mult

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(title, "MACD", "RSI"),
    )

    fig.add_trace(
        go.Candlestick(
            x=tail["timestamp"],
            open=tail["open"],
            high=tail["high"],
            low=tail["low"],
            close=tail["close"],
            name="OHLC",
        ),
        row=1,
        col=1,
    )

    if "ema_20" in df.columns:
        fig.add_trace(
            go.Scatter(x=tail["timestamp"], y=tail["ema_20"] * mult, name="EMA20", mode="lines"),
            row=1,
            col=1,
        )
    if "ema_50" in df.columns:
        fig.add_trace(
            go.Scatter(x=tail["timestamp"], y=tail["ema_50"] * mult, name="EMA50", mode="lines"),
            row=1,
            col=1,
        )
    if {"bb_bbh", "bb_bbl"}.issubset(df.columns):
        fig.add_trace(
            go.Scatter(x=tail["timestamp"], y=tail["bb_bbh"] * mult, name="BB Upper", mode="lines", line=dict(dash="dot")),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=tail["timestamp"], y=tail["bb_bbl"] * mult, name="BB Lower", mode="lines", line=dict(dash="dot")),
            row=1,
            col=1,
        )
        if "bb_bbm" in df.columns:
            fig.add_trace(
                go.Scatter(x=tail["timestamp"], y=tail["bb_bbm"] * mult, name="BB Mid", mode="lines", line=dict(dash="dot")),
                row=1,
                col=1,
            )

    if tp is not None:
        fig.add_hline(y=tp * mult, line=dict(color="#2ca02c", dash="dash"), annotation_text=f"TP {tp * mult:,.2f}", row=1, col=1)
    if sl is not None:
        fig.add_hline(y=sl * mult, line=dict(color="#d62728", dash="dash"), annotation_text=f"SL {sl * mult:,.2f}", row=1, col=1)

    if {"macd", "macd_signal"}.issubset(df.columns):
        tail_macd = df.tail(lookback)
        if "macd_diff" in df.columns:
            fig.add_trace(
                go.Bar(x=tail_macd["timestamp"], y=tail_macd["macd_diff"], name="MACD Diff", marker_color="#8c564b"),
                row=2,
                col=1,
            )
        fig.add_trace(
            go.Scatter(x=tail_macd["timestamp"], y=tail_macd["macd"], name="MACD", mode="lines"),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=tail_macd["timestamp"], y=tail_macd["macd_signal"], name="Signal", mode="lines"),
            row=2,
            col=1,
        )

    if "rsi_14" in df.columns:
        tail_rsi = df.tail(lookback)
        fig.add_trace(
            go.Scatter(x=tail_rsi["timestamp"], y=tail_rsi["rsi_14"], name="RSI", mode="lines", line=dict(color="#1f77b4")),
            row=3,
            col=1,
        )
        fig.add_hline(y=70, line=dict(color="#d62728", dash="dot"), row=3, col=1)
        fig.add_hline(y=30, line=dict(color="#2ca02c", dash="dot"), row=3, col=1)
        fig.update_yaxes(range=[0, 100], row=3, col=1)

    fig.update_layout(
        height=700,
        margin=dict(l=10, r=10, t=40, b=20),
        showlegend=True,
        xaxis_rangeslider_visible=False,
    )
    return fig


def render_panel(asset_key: str, cfg: Dict[str, Any], lookback: int) -> None:
    st.subheader(cfg["title"])

    c1, c2, c3 = st.columns([1, 1, 1])
    features_csv = c1.text_input(
        f"{asset_key} features CSV", value=cfg["features_csv"], key=f"{asset_key}_csv"
    )
    display_mult = c2.number_input(
        f"{asset_key} display multiplier", value=float(cfg["display_mult"]), step=0.01, key=f"{asset_key}_mult"
    )
    use_api_asset = c3.checkbox(
        f"Use API asset={asset_key}", value=True, key=f"{asset_key}_api"
    )

    signal: Dict[str, Any] | None = None
    latency_ms: float | None = None
    try:
        if use_api_asset:
            signal, latency_ms = api_get("/signal", params={"asset": asset_key})
        else:
            signal, latency_ms = api_get(
                "/signal",
                params={
                    "features_csv": features_csv,
                    "model_path": "models/xgb_barrier.joblib",
                },
            )
    except Exception as exc:
        st.error(f"Signal error: {exc}")

    try:
        if use_api_asset:
            df, bars_latency = api_bars(asset_key, max(lookback, 300))
        else:
            df = load_df(features_csv)
            bars_latency = None
    except Exception as exc:
        st.error(f"Bars error: {exc}")
        return

    sym, src = last_symbol_source(df)
    synthetic = sym.upper().endswith("_SYN") or src.lower() == "synthetic"

    m1, m2, m3, m4, m5, m6, m7, m8 = st.columns(8)
    latest = df.iloc[-1]
    disp_close = latest["close"] * display_mult
    m1.metric("Timestamp", str(latest["timestamp"]))
    m2.metric("Close (disp.)", f"{disp_close:,.2f}")

    if len(df) > 1:
        candle_delta = latest["timestamp"] - df.iloc[-2]["timestamp"]
    else:
        candle_delta = pd.Timedelta(0)

    if signal:
        m3.metric("Prob LONG", f"{signal['p_long']:.3f}")
        m4.metric("Prob SHORT", f"{signal['p_short']:.3f}")
        side_map = {1: "BUY", -1: "SELL", 0: "FLAT"}
        decision = side_map.get(signal.get("side"), "FLAT")
        color_map = {"BUY": "#1f8a4c", "SELL": "#c0392b", "FLAT": "#d4ac0d"}
        decision_color = color_map.get(decision, "#7f8c8d")
        m5.markdown(
            f"<div style='padding:0.6rem;border-radius:6px;text-align:center;font-weight:600;"
            f"color:#ffffff;background:{decision_color};'>Decision: {decision}</div>",
            unsafe_allow_html=True,
        )
        m6.metric("Signal ms", f"{latency_ms:.0f}" if latency_ms is not None else "—")
    else:
        m3.metric("Prob LONG", "—")
        m4.metric("Prob SHORT", "—")
        m5.markdown(
            "<div style='padding:0.6rem;border-radius:6px;text-align:center;font-weight:600;"
            "color:#ffffff;background:#7f8c8d;'>Decision: —</div>",
            unsafe_allow_html=True,
        )
        m6.metric("Signal ms", "—")

    if bars_latency is not None and use_api_asset:
        m7.metric("Bars ms", f"{bars_latency:.0f}")
    else:
        m7.metric("Bars ms", "—")

    m8.metric("Candle Δ", format_timedelta(candle_delta))

    if use_api_asset and not df.empty:
        tf_sec = 300 if asset_key == "XAU" else 60
        remain = seconds_to_next_bar(latest["timestamp"], tf_sec)
        tf_label = f"{tf_sec // 60}m" if tf_sec >= 60 else f"{tf_sec}s"
        st.caption(f"{asset_key} cadence: {tf_label} — {remain}s to next bar close")

    b1, b2 = st.columns(2)
    b1.caption(f"symbol: **{sym}** · source: **{src}**")
    if synthetic:
        b2.error("SYNTHETIC DATA (fallback)")
    else:
        b2.caption("Data source OK")

    tp_val = signal.get("tp") if signal else None
    sl_val = signal.get("sl") if signal else None
    fig = make_dashboard_fig(
        df,
        lookback=lookback,
        mult=display_mult,
        tp=tp_val,
        sl=sl_val,
        title=f"{asset_key} — last {lookback} bars",
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander(f"{asset_key} — raw signal"):
        st.json(signal or {})


# ---------- UI ----------
st.set_page_config(page_title="Signals (XAU + BTC)", layout="wide")
st.title("Gold & BTC Signals")
last_refresh = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
st.caption(f"Last refresh: {last_refresh}")

with st.sidebar:
    st.subheader("Global settings")
    refresh_sec = st.number_input(
        "Auto-refresh (seconds)", min_value=0, step=1, value=15, help="0 disables auto-refresh"
    )
    lookback = st.number_input("Candles lookback (bars)", min_value=50, step=50, value=300)
    API_URL = st.text_input("API URL", value=API_URL)
    API_TOKEN = st.text_input("API Token (X-Token header)", value=API_TOKEN, type="password")
    st.caption("Tip: set API_URL & API_TOKEN via environment variables for persistence.")
    if refresh_sec == 0:
        st.warning("Auto-refresh is off. Increase interval for live updates.")
    else:
        st.caption(f"Auto-refresh interval: {refresh_sec}s")

try:
    from streamlit_extras.st_autorefresh import st_autorefresh  # type: ignore
except Exception:  # pragma: no cover
    st_autorefresh = None

if refresh_sec > 0:
    if st_autorefresh is not None:
        st_autorefresh(interval=refresh_sec * 1000, key="autorefresh")
    else:
        st.markdown(
            f"<script>setTimeout(function(){{window.location.reload();}}, {int(refresh_sec * 1000)});</script>",
            unsafe_allow_html=True,
        )

left_col, right_col = st.columns(2, gap="large")
with left_col:
    render_panel("XAU", DEFAULTS["XAU"], lookback)
with right_col:
    render_panel("BTC", DEFAULTS["BTC"], lookback)

st.caption(
    "Charts show display-scaled prices only. Backtests and live signals always use raw feature values."
)
