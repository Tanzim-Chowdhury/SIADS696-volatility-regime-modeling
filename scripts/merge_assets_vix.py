#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge cleaned asset OHLCV with cleaned VIX by date.")
    p.add_argument("--assets", required=True, help="Clean assets parquet (data/clean/assets_clean.parquet)")
    p.add_argument("--vix", required=True, help="Clean VIX parquet (data/clean/vix_clean.parquet)")
    p.add_argument("--out", required=True, help="Output parquet (data/interim/ohlcv_with_vix.parquet)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    assets = pd.read_parquet(args.assets)
    vix = pd.read_parquet(args.vix)

    assets["date"] = pd.to_datetime(assets["date"]).dt.tz_localize(None)
    vix["date"] = pd.to_datetime(vix["date"]).dt.tz_localize(None)

    # Keep only VIX rows (defensive)
    vix = vix[vix["ticker"].astype(str).isin(["^VIX", "VIX"])].copy()
    vix = vix.sort_values("date")

    # Rename to avoid collisions
    vix = vix.rename(columns={"close": "vix_close", "volume": "vix_volume"})

    # VIX-derived features (market context)
    vix["vix_log_return"] = np.log(vix["vix_close"]).diff()
    vix["vix_rv_10"] = vix["vix_log_return"].rolling(10, min_periods=10).std()
    vix["vix_rv_20"] = vix["vix_log_return"].rolling(20, min_periods=20).std()

    vix = vix[["date", "vix_close", "vix_volume", "vix_log_return", "vix_rv_10", "vix_rv_20"]]

    assets = assets.sort_values(["ticker", "date"]).reset_index(drop=True)
    merged = assets.merge(vix, on="date", how="left")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)

    print(f"Saved merged dataset to: {out_path}")
    print("Rows:", len(merged))
    print("Missing VIX rows:", int(merged["vix_close"].isna().sum()))
    print("Date range:", merged["date"].min(), "→", merged["date"].max())


if __name__ == "__main__":
    main()
