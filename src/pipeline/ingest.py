from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class FetchSpec:
    tickers: list[str]
    start: str
    end: Optional[str] = None
    interval: str = "1d"
    auto_adjust: bool = False


def fetch_ohlcv(spec: FetchSpec) -> pd.DataFrame:
    raw = yf.download(
        tickers=spec.tickers,
        start=spec.start,
        end=spec.end,
        interval=spec.interval,
        group_by="column",
        auto_adjust=spec.auto_adjust,
        progress=False,
        threads=True,
    )

    if raw.empty:
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"])

    if isinstance(raw.columns, pd.MultiIndex):
        df = raw.stack(level=1).reset_index()
        # normalize ticker column name
        for cand in ["ticker", "Ticker", "Symbols", "level_1"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "ticker"})
                break
        if "ticker" not in df.columns:
            raise KeyError(f"Could not locate ticker column after stacking. Columns={list(df.columns)}")
    else:
        df = raw.reset_index()
        df["ticker"] = spec.tickers[0]

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

    if "adj_close" not in df.columns:
        df["adj_close"] = pd.NA

    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["ticker"] = df["ticker"].astype(str)

    keep = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    df = df[keep].sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    required = ["date", "ticker", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out["ticker"] = out["ticker"].astype(str).str.strip()
    out = out.sort_values(["ticker", "date"]).drop_duplicates(subset=["ticker", "date"], keep="last")

    # basic validity
    for c in ["open", "high", "low", "close"]:
        out = out[out[c].notna() & (out[c] > 0)]
    out = out[out["volume"].notna() & (out["volume"] >= 0)]
    out = out[out["high"] >= out["low"]]

    keep = ["date", "ticker", "open", "high", "low", "close", "volume"]
    if "adj_close" in out.columns:
        keep.insert(6, "adj_close")
    return out[keep].reset_index(drop=True)


def merge_assets_with_vix(assets: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    a = assets.copy()
    v = vix.copy()

    a["date"] = pd.to_datetime(a["date"]).dt.tz_localize(None)
    v["date"] = pd.to_datetime(v["date"]).dt.tz_localize(None)

    v = v[v["ticker"].astype(str).isin(["^VIX", "VIX"])].copy().sort_values("date")
    v = v.rename(columns={"close": "vix_close", "volume": "vix_volume"})

    v["vix_log_return"] = np.log(v["vix_close"]).diff()
    v["vix_rv_10"] = v["vix_log_return"].rolling(10, min_periods=10).std()
    v["vix_rv_20"] = v["vix_log_return"].rolling(20, min_periods=20).std()

    v = v[["date", "vix_close", "vix_volume", "vix_log_return", "vix_rv_10", "vix_rv_20"]]
    a = a.sort_values(["ticker", "date"]).reset_index(drop=True)

    return a.merge(v, on="date", how="left")