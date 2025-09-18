from __future__ import annotations
import json
import pathlib
import numpy as np
import pandas as pd

FEATURE_EXCLUDE = {
    "timestamp",
    "source",
    "symbol",
    "timeframe",
    "y_next3_sign",
    "y_atr",
}

def default_feature_cols(df: pd.DataFrame):
    cols = [c for c in df.columns if c not in FEATURE_EXCLUDE and df[c].dtype != "O"]
    return cols

def save_json(obj, path):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)

