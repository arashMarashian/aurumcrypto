from __future__ import annotations
import argparse
import pathlib
import pandas as pd

from common import DATA_DIR, normalize_ohlcv_df, save_csv_parquet


def _synthetic_ohlcv(limit: int, timeframe: str) -> pd.DataFrame:
    """Generate deterministic synthetic OHLCV data for offline usage."""
    freq = pd.to_timedelta(timeframe)
    end = pd.Timestamp.utcnow().floor(freq)
    idx = pd.date_range(end=end, periods=limit, freq=freq)
    trend = pd.Series(range(limit), dtype=float)
    open_ = 100.0 + trend * 0.5
    close = open_ + 0.1
    high = open_ + 0.3
    low = open_ - 0.3
    volume = 1000 + trend * 2
    return pd.DataFrame({
        "timestamp": (idx.view("int64") // 10**6),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def _synthetic_yf(limit: int, interval: str) -> pd.DataFrame:
    synthetic = _synthetic_ohlcv(limit, interval)
    synthetic["Datetime"] = pd.to_datetime(synthetic["timestamp"], unit="ms", utc=True)
    synthetic = synthetic.drop(columns=["timestamp"])
    cols = ["Datetime", "open", "high", "low", "close", "volume"]
    synthetic = synthetic[cols]
    synthetic.attrs["synthetic"] = True
    return synthetic


def fetch_ccxt_ohlcv(symbol: str, timeframe: str = "1m", limit: int = 1000, exchange_name: str = "binance") -> pd.DataFrame:
    try:
        import ccxt
    except ImportError:
        print("[crypto] WARNING: ccxt not installed; generating synthetic OHLCV data.")
        return _synthetic_ohlcv(limit, timeframe)
    exchange = getattr(ccxt, exchange_name)()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    return pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])


def fetch_yf(symbol: str, interval: str = "5m", period: str = "30d") -> tuple[pd.DataFrame, bool]:
    try:
        import yfinance as yf
    except ImportError:
        print("[gold] WARNING: yfinance not installed; generating synthetic OHLCV data.")
        return _synthetic_yf(500, interval), True

    df = yf.download(symbol, interval=interval, period=period, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.get_level_values(0)
        except Exception:
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    if df.empty:
        print("[gold] WARNING: empty dataframe from yfinance; generating synthetic OHLCV data.")
        return _synthetic_yf(500, interval), True

    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.get_level_values(0)
        except Exception:
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume", "Adj Close": "adj_close"}, inplace=True)
    df.attrs["synthetic"] = False
    return df, False


def main():
    parser = argparse.ArgumentParser(description="Unified data ingestion for BTC (ccxt) and Gold (yfinance).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ap_crypto = sub.add_parser("crypto", help="Fetch crypto OHLCV via ccxt")
    ap_crypto.add_argument("--symbol", default="BTC/USDT", help="ccxt symbol, e.g., BTC/USDT")
    ap_crypto.add_argument("--exchange", default="binance", help="ccxt exchange id, e.g., binance")
    ap_crypto.add_argument("--timeframe", default="1m", help="ccxt timeframe, e.g., 1m,5m,15m")
    ap_crypto.add_argument("--limit", type=int, default=1000, help="number of candles")
    ap_crypto.add_argument("--out", default=None, help="output base path (without extension)")

    ap_gold = sub.add_parser("gold", help="Fetch gold proxy via yfinance")
    ap_gold.add_argument("--ticker", default="GC=F", help="Yahoo ticker, e.g., GC=F or GLD")
    ap_gold.add_argument("--interval", default="5m", help="yfinance interval: 1m,2m,5m,15m,30m,60m")
    ap_gold.add_argument("--period", default="30d", help="yfinance period: 7d,30d,60d, etc.")
    ap_gold.add_argument("--out", default=None, help="output base path (without extension)")

    args = parser.parse_args()

    if args.cmd == "crypto":
        raw = fetch_ccxt_ohlcv(args.symbol, timeframe=args.timeframe, limit=args.limit, exchange_name=args.exchange)
        norm = normalize_ohlcv_df(raw, source=args.exchange, symbol=args.symbol.replace("/", ""), timeframe=args.timeframe)
        base = pathlib.Path(args.out) if args.out else DATA_DIR / f"{args.symbol.replace('/', '')}_{args.timeframe}"
        csv_path, pq_path = save_csv_parquet(norm, base)
        print(f"[crypto] rows={len(norm)} range={norm['timestamp'].min()} → {norm['timestamp'].max()}")
        print("[crypto] written:", csv_path, pq_path if pq_path else "")
        if not norm.empty:
            print(norm.head(3).to_string(index=False))
            print("...")
            print(norm.tail(3).to_string(index=False))
        else:
            print("[crypto] WARNING: normalized dataframe is empty.")

    elif args.cmd == "gold":
        raw, synthetic = fetch_yf(args.ticker, interval=args.interval, period=args.period)
        if raw.empty:
            print("[gold] WARNING: empty dataframe from source; nothing to normalize.")
            norm = raw
        else:
            if synthetic:
                norm = normalize_ohlcv_df(raw, source="synthetic", symbol="XAU_SYN", timeframe=args.interval)
            else:
                norm = normalize_ohlcv_df(raw, source="yfinance", symbol=args.ticker, timeframe=args.interval)
        base = pathlib.Path(args.out) if args.out else DATA_DIR / f"{args.ticker.replace('=', '').replace('^', '')}_{args.interval}"
        csv_path, pq_path = save_csv_parquet(norm, base)
        print(f"[gold] rows={len(norm)} range={norm['timestamp'].min() if not norm.empty else 'NaT'} → {norm['timestamp'].max() if not norm.empty else 'NaT'}")
        print("[gold] written:", csv_path, pq_path if pq_path else "")
        if not norm.empty:
            print(norm.head(3).to_string(index=False))
            print("...")
            print(norm.tail(3).to_string(index=False))
        else:
            print("[gold] WARNING: normalized dataframe is empty.")


if __name__ == "__main__":
    main()
