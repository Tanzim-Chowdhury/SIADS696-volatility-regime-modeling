#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build leakage-safe volatility features + forward label from merged OHLCV+VIX.")
    p.add_argument("--infile", required=True, help="Merged parquet (data/interim/ohlcv_with_vix.parquet)")
    p.add_argument("--out", required=True, help="Output parquet (data/processed/vol_features.parquet)")
    p.add_argument("--horizon", type=int, default=5, help="Forward horizon in trading days (default 5)")
    p.add_argument("--train-end", required=True, help="Train end date YYYY-MM-DD (for regime quantiles)")
    p.add_argument("--n-regimes", type=int, default=3, help="Number of regimes (default 3)")
    return p.parse_args()


def realized_vol(returns: pd.Series, window: int, annualization: int = 252) -> pd.Series:
    return np.sqrt(annualization) * returns.rolling(window=window, min_periods=window).std()


def forward_realized_vol(returns: pd.Series, horizon: int, annualization: int = 252) -> pd.Series:
    # label at t uses returns t+1..t+horizon
    return np.sqrt(annualization) * returns.rolling(horizon, min_periods=horizon).std().shift(-horizon)


def main() -> None:
    args = parse_args()

    df = pd.read_parquet(args.infile)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Per-ticker features
    out_parts = []
    for ticker, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date").copy()

        # Core return feature
        g["log_return"] = np.log(g["close"]).diff()

        # Realized-vol features (past)
        for w in (10, 20, 60):
            g[f"rv_{w}"] = realized_vol(g["log_return"], w)

        # Volume features (past)
        for w in (10, 20):
            g[f"vol_mean_{w}"] = g["volume"].rolling(w, min_periods=w).mean()
            g[f"vol_std_{w}"] = g["volume"].rolling(w, min_periods=w).std()

        # Forward label (future)
        g[f"fwd_rv_{args.horizon}"] = forward_realized_vol(g["log_return"], args.horizon)

        out_parts.append(g)

    feat = pd.concat(out_parts, axis=0, ignore_index=True)

    # Drop rows missing rolling features or label (leakage-safe)
    needed = [
        "log_return", "rv_10", "rv_20", "rv_60",
        f"fwd_rv_{args.horizon}",
        "vix_close", "vix_rv_10", "vix_rv_20",
    ]
    feat = feat.dropna(subset=[c for c in needed if c in feat.columns]).reset_index(drop=True)

    # Regime labels using ONLY training period quantiles (leakage-safe)
    train_end = pd.to_datetime(args.train_end)
    ycol = f"fwd_rv_{args.horizon}"

    train_slice = feat[feat["date"] <= train_end]
    edges = np.quantile(train_slice[ycol].values, np.linspace(0, 1, args.n_regimes + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        raise ValueError("Not enough label variation to create regimes. Use longer history or different horizon.")

    feat["regime"] = pd.cut(feat[ycol], bins=edges, include_lowest=True, labels=False)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(out_path, index=False)

    print(f"Saved features: {out_path}")
    print("Rows:", len(feat))
    print("Date range:", feat["date"].min(), "→", feat["date"].max())
    print("Regime counts:\n", feat["regime"].value_counts(dropna=False).sort_index())


if __name__ == "__main__":
    main()
