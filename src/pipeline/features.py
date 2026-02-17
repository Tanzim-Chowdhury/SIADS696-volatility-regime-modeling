from __future__ import annotations

import numpy as np
import pandas as pd


def realized_vol(returns: pd.Series, window: int, annualization: int = 252) -> pd.Series:
    return np.sqrt(annualization) * returns.rolling(window=window, min_periods=window).std()


def forward_realized_vol(returns: pd.Series, horizon: int, annualization: int = 252) -> pd.Series:
    # label at t uses returns t+1..t+horizon
    return np.sqrt(annualization) * returns.rolling(horizon, min_periods=horizon).std().shift(-horizon)


def build_features_and_labels(
    merged_df: pd.DataFrame,
    horizon_days: int = 5,
    rv_windows: tuple[int, ...] = (10, 20, 60),
    vol_windows: tuple[int, ...] = (10, 20),
    train_end: str = "2023-12-29",
    n_regimes: int = 3,
) -> pd.DataFrame:
    d = merged_df.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None)
    d = d.sort_values(["ticker", "date"]).reset_index(drop=True)

    parts = []
    for _, g in d.groupby("ticker", sort=False):
        g = g.sort_values("date").copy()
        g["log_return"] = np.log(g["close"]).diff()

        for k in (1, 2, 3, 5):
            g[f"log_return_lag{k}"] = g["log_return"].shift(k)

        # VIX lag features (exogenous shock memory) - same for all tickers after merge
        if "vix_log_return" in g.columns:
            for k in (1, 2, 3):
                g[f"vix_log_return_lag{k}"] = g["vix_log_return"].shift(k)

        for w in rv_windows:
            g[f"rv_{w}"] = realized_vol(g["log_return"], w)

        for w in vol_windows:
            g[f"vol_mean_{w}"] = g["volume"].rolling(w, min_periods=w).mean()
            g[f"vol_std_{w}"] = g["volume"].rolling(w, min_periods=w).std()

        g[f"fwd_rv_{horizon_days}"] = forward_realized_vol(g["log_return"], horizon_days)
        parts.append(g)

    feat = pd.concat(parts, ignore_index=True)

    needed = (
        ["log_return"]
        + [f"rv_{w}" for w in rv_windows]
        + [f"vol_mean_{w}" for w in vol_windows]
        + [f"vol_std_{w}" for w in vol_windows]
        + [f"fwd_rv_{horizon_days}"]
        + ["vix_close", "vix_rv_10", "vix_rv_20"]
        + [f"log_return_lag{k}" for k in (1, 2, 3, 5)]
        + [f"vix_log_return_lag{k}" for k in (1, 2, 3)]
    )
    needed = [c for c in needed if c in feat.columns]
    feat = feat.dropna(subset=needed).reset_index(drop=True)

    # leakage-safe regime bins: compute edges using only training dates
    train_end_dt = pd.to_datetime(train_end)
    ycol = f"fwd_rv_{horizon_days}"
    train_slice = feat[feat["date"] <= train_end_dt]
    edges = np.quantile(train_slice[ycol].values, np.linspace(0, 1, n_regimes + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        raise ValueError("Not enough label variation to create regimes. Try longer history or different horizon.")

    feat["regime"] = pd.cut(feat[ycol], bins=edges, include_lowest=True, labels=False)
    return feat