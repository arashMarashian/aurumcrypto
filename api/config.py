from __future__ import annotations
import os
import pathlib


def _load_env_file():
    env_path = pathlib.Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key not in os.environ:
            os.environ[key] = value.strip()


_load_env_file()


def get_api_token() -> str:
    return os.getenv("API_TOKEN", "").strip()


# Default asset routing (adjust paths as you like)
ASSETS = {
    "XAU": {
        "features": "data/xauusd_5m_atr.csv",
        "model": "models/xgb_barrier.joblib",
        "meta": "models/xgb_barrier_meta.json",
    },
    "BTC": {
        "features": "data/btcusdt_1m_atr.csv",
        "model":    "models/xgb_barrier_btc.joblib",
        "meta":     "models/xgb_barrier_btc_meta.json",
    },
}
