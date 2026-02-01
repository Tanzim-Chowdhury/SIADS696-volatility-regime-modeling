#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.features.volatility import (
    VolatilityConfig,
    add_volatility_features,
    add_regime_labels,
    drop_leakage_na_rows,
    save_parquet,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build volatility features + forward labels from OHLCV parquet.")
    p.add_argument("--infile", required=True, help="Path to input OHLCV parquet (from data/raw).")
    p.add_argument("--outfile", required=True, help="Path to output parquet (e.g., data/processed/vol_features.parquet).")
    p.add_argument("--horizon", type=int, default=5, help="Forward label horizon in trading days (default 5).")
    p.add_argument("--train-end", required=True, help="Train end date YYYY-MM-DD (used for leakage-safe regime bins).")
    p.add_argument("--n-regimes", type=int, default=3, help="Number of regimes (default 3).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    infile = Path(args.infile)
    outfile = Path(args.outfile)

    df = pd.read_parquet(infile)

    cfg = VolatilityConfig(horizon_days=args.horizon)

    # Features + continuous label
    feat_df = add_volatility_features(df, cfg=cfg)

    # Drop rows where rolling windows / forward labels aren't available
    feat_df = drop_leakage_na_rows(feat_df, cfg=cfg)

    label_col = f"fwd_rv_{cfg.horizon_days}"
    feat_df = add_regime_labels(
        feat_df,
        label_col=label_col,
        train_end=args.train_end,
        n_regimes=args.n_regimes,
    )

    save_parquet(feat_df, outfile)
    print(f"Saved features to: {outfile}")
    print("Columns:", list(feat_df.columns))
    print("Date range:", feat_df["date"].min(), "→", feat_df["date"].max())
    print("Regime counts:\n", feat_df["regime"].value_counts(dropna=False).sort_index())


if __name__ == "__main__":
    main()
