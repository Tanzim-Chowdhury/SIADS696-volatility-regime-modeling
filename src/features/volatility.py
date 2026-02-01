from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VolatilityConfig:
    # horizon for supervised label (future realized vol)
    horizon_days: int = 5

    # rolling windows for features (past realized vol)
    rv_windows: tuple[int, ...] = (10, 20, 60)

    # for volume features
    vol_windows: tuple[int, ...] = (10, 20)

    # annualization factor for daily data
    annualization: int = 252

    # number of regime bins for classification
    n_regimes: int = 3  # low / mid / high


def _realized_vol_from_returns(r: pd.Series, window: int, annualization: int) -> pd.Series:
    """
    Realized volatility: sqrt(annualization) * rolling std of returns.
    """
    return np.sqrt(annualization) * r.rolling(window=window, min_periods=window).std()


def _forward_realized_vol(r: pd.Series, horizon: int, annualization: int) -> pd.Series:
    """
    Forward-looking realized volatility label:
      vol_{t->t+h} computed from returns at t+1..t+h.

    We compute rolling std over horizon, then shift it backward so label at time t
    uses future returns only.
    """
    future_std = r.rolling(window=horizon, min_periods=horizon).std()
    # future_std at time t represents std over [t-h+1 .. t], so shift backward
    # so label at t uses returns t+1..t+h:
    return np.sqrt(annualization) * future_std.shift(-horizon)


def add_volatility_features(
    df: pd.DataFrame,
    cfg: VolatilityConfig,
) -> pd.DataFrame:
    """
    Input df must have columns: date, ticker, close, volume
    Output includes:
      - log_return
      - realized vol features: rv_{window}
      - volume features: vol_mean_{w}, vol_std_{w}
      - forward label: fwd_rv_{horizon}
    """
    required = {"date", "ticker", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)

    # group-wise calculations (per ticker)
    parts = []
    for ticker, g in out.groupby("ticker", sort=False):
        g = g.sort_values("date").copy()

        # log returns
        g["log_return"] = np.log(g["close"]).diff()

        # realized vol features (past)
        for w in cfg.rv_windows:
            g[f"rv_{w}"] = _realized_vol_from_returns(g["log_return"], window=w, annualization=cfg.annualization)

        # volume features
        for w in cfg.vol_windows:
            g[f"vol_mean_{w}"] = g["volume"].rolling(window=w, min_periods=w).mean()
            g[f"vol_std_{w}"] = g["volume"].rolling(window=w, min_periods=w).std()

        # forward label (future)
        g[f"fwd_rv_{cfg.horizon_days}"] = _forward_realized_vol(
            g["log_return"], horizon=cfg.horizon_days, annualization=cfg.annualization
        )

        parts.append(g)

    out = pd.concat(parts, axis=0, ignore_index=True)
    return out


def time_split(
    df: pd.DataFrame,
    train_end: str,
    val_end: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by date:
      train: date <= train_end
      val:   (train_end, val_end]  (if val_end provided)
      test:  date > val_end (or > train_end if val_end is None)

    Returns (train, val, test).
    """
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None)

    train_end_dt = pd.to_datetime(train_end)
    if val_end is not None:
        val_end_dt = pd.to_datetime(val_end)
    else:
        val_end_dt = None

    train = d[d["date"] <= train_end_dt].copy()

    if val_end_dt is not None:
        val = d[(d["date"] > train_end_dt) & (d["date"] <= val_end_dt)].copy()
        test = d[d["date"] > val_end_dt].copy()
    else:
        val = d.iloc[0:0].copy()
        test = d[d["date"] > train_end_dt].copy()

    return train, val, test


def add_regime_labels(
    df_with_labels: pd.DataFrame,
    label_col: str,
    train_end: str,
    n_regimes: int = 3,
) -> pd.DataFrame:
    """
    Create discrete regime labels from a continuous label column (e.g., fwd_rv_5).

    IMPORTANT (leakage-safe):
    - quantile cutoffs are computed using ONLY training dates (<= train_end)
    - then applied to all rows
    """
    d = df_with_labels.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None)

    train_end_dt = pd.to_datetime(train_end)
    train_slice = d[d["date"] <= train_end_dt]

    if label_col not in d.columns:
        raise ValueError(f"Label column '{label_col}' not found.")

    # compute quantile bins on training slice only
    y = train_slice[label_col].dropna()
    if y.empty:
        raise ValueError("Training slice has no non-null labels. Check date range and horizon.")

    # quantile edges
    qs = np.linspace(0, 1, n_regimes + 1)
    edges = np.quantile(y.values, qs)
    # ensure strictly increasing edges (handle degenerate cases)
    edges = np.unique(edges)
    if len(edges) < 3:
        raise ValueError("Not enough label variation to create regimes. Try different horizon or longer history.")

    # pd.cut needs monotonically increasing bin edges
    d["regime"] = pd.cut(
        d[label_col],
        bins=edges,
        include_lowest=True,
        labels=False,
    )

    # optional: make regimes 0..(k-1)
    return d


def drop_leakage_na_rows(
    df: pd.DataFrame,
    cfg: VolatilityConfig,
) -> pd.DataFrame:
    """
    Remove rows where we can't compute required rolling features or the forward label.
    """
    cols_needed = ["log_return"] + [f"rv_{w}" for w in cfg.rv_windows] + [f"fwd_rv_{cfg.horizon_days}"]
    cols_needed += [f"vol_mean_{w}" for w in cfg.vol_windows] + [f"vol_std_{w}" for w in cfg.vol_windows]

    existing = [c for c in cols_needed if c in df.columns]
    cleaned = df.dropna(subset=existing).copy()
    return cleaned.reset_index(drop=True)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
