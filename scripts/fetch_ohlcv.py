#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf


def _slug_ticker(t: str) -> str:
    # Make tickers filename-safe: "^VIX" -> "VIX"
    return t.strip().replace("^", "").replace("/", "_").replace(" ", "")


def _tickers_slug(tickers: list[str]) -> str:
    return "-".join(_slug_ticker(t) for t in tickers)


def _default_outpath(out_dir: Path, prefix: str, tickers: list[str], start: str, end: Optional[str], interval: str) -> Path:
    end_part = end if end else "latest"
    name = f"{prefix}{_tickers_slug(tickers)}_{start}_{end_part}_{interval}.parquet"
    return out_dir / name


def fetch_ohlcv(
    tickers: list[str],
    start: str,
    end: Optional[str],
    interval: str,
    auto_adjust: bool,
) -> pd.DataFrame:
    """
    Fetch OHLCV from yfinance. Returns standardized long-form:
      date, ticker, open, high, low, close, adj_close, volume
    """
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        group_by="column",
        auto_adjust=auto_adjust,
        progress=False,
        threads=True,
    )

    if raw.empty:
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"])

    # MultiIndex columns for multiple tickers; single index for one ticker
    if isinstance(raw.columns, pd.MultiIndex):
    # Stack to long form; the stacked level name varies across yfinance versions
        df = raw.stack(level=1).reset_index()

    # yfinance may name the ticker level differently; normalize it to "ticker"
    possible = ["ticker", "Ticker", "Symbols", "level_1"]
    ticker_col = next((c for c in possible if c in df.columns), None)
    if ticker_col is None:
        raise KeyError(f"Could not find ticker column after stacking. Columns: {list(df.columns)}")

    if ticker_col != "ticker":
        df = df.rename(columns={ticker_col: "ticker"})
    else:
        # Single ticker output
        df = raw.reset_index()
        df["ticker"] = tickers[0]


    df = df.rename(columns={
        "Date": "date",
        "Datetime": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    })

    # Ensure adj_close exists
    if "adj_close" not in df.columns:
        df["adj_close"] = pd.NA

    # Normalize date + types
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["ticker"] = df["ticker"].astype(str)

    # Keep canonical order
    keep = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    df = df[keep].sort_values(["ticker", "date"]).reset_index(drop=True)

    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch OHLCV from Yahoo Finance via yfinance and save as parquet.")
    p.add_argument("--tickers", nargs="+", required=True, help="Tickers, e.g. SPY QQQ AAPL or ^VIX")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", default=None, help="End date YYYY-MM-DD (optional)")
    p.add_argument("--interval", default="1d", help="Interval (default 1d). Examples: 1d, 1h")
    p.add_argument("--auto-adjust", action="store_true", help="Use adjusted prices (splits/dividends).")
    p.add_argument("--out-dir", default="data/raw", help="Output directory (default data/raw)")
    p.add_argument("--prefix", default="ohlcv_", help="Filename prefix (e.g., assets_ or vix_)")
    p.add_argument("--out", default=None, help="Optional full output path (overrides --out-dir/--prefix)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.out) if args.out else _default_outpath(
        out_dir=out_dir,
        prefix=args.prefix,
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        interval=args.interval,
    )

    df = fetch_ohlcv(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        interval=args.interval,
        auto_adjust=args.auto_adjust,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    print(f"Saved: {out_path}")
    if not df.empty:
        print(f"Rows: {len(df):,} | Date range: {df['date'].min().date()} → {df['date'].max().date()} | Tickers: {sorted(df['ticker'].unique())}")


if __name__ == "__main__":
    main()
