#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean OHLCV parquet (dedupe, validate, normalize).")
    p.add_argument("--infile", required=True, help="Input parquet path (data/raw/*.parquet)")
    p.add_argument("--out", required=True, help="Output parquet path (data/clean/*.parquet)")
    p.add_argument("--keep-adj-close", action="store_true", help="Keep adj_close column (default keeps it anyway)")
    return p.parse_args()


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    required = ["date", "ticker", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()

    # Normalize date
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)

    # Standardize ticker
    out["ticker"] = out["ticker"].astype(str).str.strip()

    # Sort
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Drop duplicate keys (keep last)
    out = out.drop_duplicates(subset=["ticker", "date"], keep="last")

    # Basic validity checks
    # Prices should be > 0, volume >= 0
    price_cols = ["open", "high", "low", "close"]
    for c in price_cols:
        out = out[out[c].notna()]
        out = out[out[c] > 0]

    out = out[out["volume"].notna()]
    out = out[out["volume"] >= 0]

    # Optional: enforce high >= low (drop invalid bars)
    out = out[out["high"] >= out["low"]]

    # Keep only expected columns (plus adj_close if present)
    keep = ["date", "ticker", "open", "high", "low", "close", "volume"]
    if "adj_close" in out.columns:
        keep.insert(6, "adj_close")  # keep near close
    out = out[keep].reset_index(drop=True)

    return out


def main() -> None:
    args = parse_args()
    infile = Path(args.infile)
    outfile = Path(args.out)

    df = pd.read_parquet(infile)
    cleaned = clean_ohlcv(df)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(outfile, index=False)

    print(f"Saved cleaned OHLCV to: {outfile}")
    print("Rows:", len(cleaned))
    print("Date range:", cleaned["date"].min(), "→", cleaned["date"].max())
    print("Tickers:", sorted(cleaned["ticker"].unique())[:10], ("..." if cleaned["ticker"].nunique() > 10 else ""))


if __name__ == "__main__":
    main()
