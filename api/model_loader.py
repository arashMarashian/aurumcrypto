from __future__ import annotations
import time
import joblib
import pathlib
from typing import Any, Dict

_CACHE: Dict[str, Any] = {}
_MTIME: Dict[str, float] = {}


def load_model(path: str):
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    mtime = p.stat().st_mtime
    if path not in _CACHE or _MTIME.get(path) != mtime:
        _CACHE[path] = joblib.load(path)
        _MTIME[path] = mtime
    return _CACHE[path]

