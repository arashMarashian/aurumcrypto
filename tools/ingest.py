from __future__ import annotations
import os, sys, argparse, pathlib, time
import pandas as pd

from common import DATA_DIR, normalize_ohlcv_df, save_csv_parquet


def _synthetic_ohlcv(limit: int, timeframe: str) -> pd.DataFrame:
    freq = pd.to_timedelta(timeframe)
    end = pd.Timestamp.utcnow().floor(freq)
    idx = pd.date_range(end=end, periods=limit, freq=freq)
    base = 100.0
    trend = pd.Series(range(limit), dtype=float)
    open_ = base + trend * 0.5
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


def fetch_ccxt_ohlcv(symbol: str, timeframe: str = "1m", limit: int = 1000, exchange_name: str = "binance") -> pd.DataFrame:
    try:
        import ccxt
    except ImportError:
        print("[crypto] WARNING: ccxt not installed; generating synthetic OHLCV data.")
        return _synthetic_ohlcv(limit, timeframe)
    ex = getattr(ccxt, exchange_name)()
    # Public OHLCV; binance supports many symbols like 'BTC/USDT'
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    return df


def fetch_yf(symbol: str, interval: str = "5m", period: str = "30d") -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        print("[gold] WARNING: yfinance not installed; generating synthetic OHLCV data.")
        synthetic = _synthetic_ohlcv(500, interval)
        synthetic.rename(columns={"timestamp": "Datetime"}, inplace=True)
        synthetic["Datetime"] = pd.to_datetime(synthetic["Datetime"], unit="ms", utc=True)
        synthetic.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}, inplace=True)
        return synthetic
    # e.g., "XAUUSD=X" or "GLD"
    df = yf.download(symbol, interval=interval, period=period, progress=False)
    if df.empty:
        return pd.DataFrame(columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])
    df = df.reset_index()  # Datetime column appears here
    df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}, inplace=True)
    return df


def main():
    ap = argparse.ArgumentParser(description="Unified data ingestion for BTC (ccxt) and Gold (yfinance).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_c = sub.add_parser("crypto", help="Fetch crypto OHLCV via ccxt")
    ap_c.add_argument("--symbol", default="BTC/USDT", help="ccxt symbol, e.g., BTC/USDT")
    ap_c.add_argument("--exchange", default="binance", help="ccxt exchange id, e.g., binance")
    ap_c.add_argument("--timeframe", default="1m", help="ccxt timeframe, e.g., 1m,5m,15m")
    ap_c.add_argument("--limit", type=int, default=1000, help="number of candles")
    ap_c.add_argument("--out", default=None, help="output base path (without extension)")

    ap_g = sub.add_parser("gold", help="Fetch gold proxy via yfinance")
    ap_g.add_argument("--ticker", default="XAUUSD=X", help="Yahoo ticker: XAUUSD=X or GLD")
    ap_g.add_argument("--interval", default="5m", help="yfinance interval: 1m,2m,5m,15m,30m,60m")
    ap_g.add_argument("--period", default="30d", help="yfinance period: 7d,30d,60d, etc.")
    ap_g.add_argument("--out", default=None, help="output base path (without extension)")

    args = ap.parse_args()

    if args.cmd == "crypto":
        raw = fetch_ccxt_ohlcv(args.symbol, timeframe=args.timeframe, limit=args.limit, exchange_name=args.exchange)
        norm = normalize_ohlcv_df(raw, source=args.exchange, symbol=args.symbol.replace("/", ""), timeframe=args.timeframe)
        base = pathlib.Path(args.out) if args.out else DATA_DIR / f"{args.symbol.replace('/', '')}_{args.timeframe}"
        csv, pq = save_csv_parquet(norm, base)
        print(f"[crypto] rows={len(norm)} range={norm['timestamp'].min()} → {norm['timestamp'].max()}")
        print("[crypto] written:", csv, pq if pq else "")
        # head/tail
        if not norm.empty:
            print(norm.head(3).to_string(index=False))
            print("...")
            print(norm.tail(3).to_string(index=False))
        else:
            print("[crypto] WARNING: normalized dataframe is empty.")

    elif args.cmd == "gold":
        raw = fetch_yf(args.ticker, interval=args.interval, period=args.period)
        if raw.empty:
            print("[gold] WARNING: empty dataframe from yfinance.")
        norm = normalize_ohlcv_df(raw, source="yfinance", symbol=args.ticker, timeframe=args.interval)
        base = pathlib.Path(args.out) if args.out else DATA_DIR / f"{args.ticker.replace('=', '').replace('^', '')}_{args.interval}"
        csv, pq = save_csv_parquet(norm, base)
        print(f"[gold] rows={len(norm)} range={norm['timestamp'].min()} → {norm['timestamp'].max()}")
        print("[gold] written:", csv, pq if pq else "")
        if not norm.empty:
            print(norm.head(3).to_string(index=False))
            print("...")
            print(norm.tail(3).to_string(index=False))
        else:
            print("[gold] WARNING: normalized dataframe is empty.")


if __name__ == "__main__":
    main()

