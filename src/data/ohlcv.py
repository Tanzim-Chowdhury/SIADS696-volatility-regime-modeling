from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import yfinance as yf

from src.config import RAW_DIR


@dataclass(frozen=True)
class OHLCVRequest:
    tickers: list[str]
    start: str  # "YYYY-MM-DD"
    end: Optional[str] = None  # "YYYY-MM-DD" or None
    interval: str = "1d"  # "1d", "1h", etc.
    auto_adjust: bool = False  # keep raw OHLC by default


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance sometimes returns:
      - single ticker: columns like ["Open","High","Low","Close","Adj Close","Volume"]
      - multi ticker: MultiIndex columns (field, ticker) or (ticker, field) depending on call.
    This function converts to a standard long format:
      date, ticker, open, high, low, close, adj_close, volume
    """
    if df.empty:
        return df

    df = df.copy()

    # If columns are MultiIndex, reshape to long
    if isinstance(df.columns, pd.MultiIndex):
        # Attempt to detect ordering
        # Common: columns = (Field, Ticker)
        if df.columns.names and ("Ticker" in df.columns.names or "Symbols" in df.columns.names):
            # Not always set; proceed with stack
            pass

        # Heuristic: if level 0 contains OHLC fields, treat as (field, ticker)
        level0 = set(map(str, df.columns.get_level_values(0)))
        ohlc_fields = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        field_first = len(level0.intersection(ohlc_fields)) >= 3

        if field_first:
            # (field, ticker)
            long_df = (
                df.stack(level=1)
                  .rename_axis(index=["date", "ticker"])
                  .reset_index()
            )
            # columns now: date, ticker, Open, High, ...
        else:
            # (ticker, field)
            long_df = (
                df.stack(level=0)
                  .rename_axis(index=["date", "ticker"])
                  .reset_index()
            )

        df = long_df

    else:
        # Single ticker: add placeholder ticker later in caller
        df = df.reset_index().rename(columns={"Date": "date"})
        # caller will add ticker column

    # Standardize column names
    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
        "Datetime": "date",
        "Date": "date",
    }
    df = df.rename(columns=rename_map)

    # Ensure required columns exist (some intervals may not provide adj_close)
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise ValueError(f"Missing expected column '{col}'. Columns: {list(df.columns)}")

    if "adj_close" not in df.columns:
        df["adj_close"] = pd.NA

    # Ensure date is datetime (timezone-naive for consistency)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

    # Order + dtypes
    df = df[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]].copy()
    df["ticker"] = df["ticker"].astype(str)

    # Sort
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    return df


def fetch_ohlcv(req: OHLCVRequest) -> pd.DataFrame:
    """
    Fetch OHLCV for one or more tickers from yfinance and return standardized long-form dataframe.
    """
    if not req.tickers:
        raise ValueError("tickers list is empty")

    # Use yf.download for multi-ticker to reduce calls.
    raw = yf.download(
        tickers=req.tickers,
        start=req.start,
        end=req.end,
        interval=req.interval,
        group_by="column",
        auto_adjust=req.auto_adjust,
        progress=False,
        threads=True,
    )

    if raw.empty:
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"])

    df = _normalize_columns(raw)

    # If single ticker and yfinance didn't include ticker col
    if "ticker" not in df.columns or df["ticker"].isna().all():
        # This path happens when df was not MultiIndex and caller didn't have ticker column
        # But in our normalize we add 'ticker' only when stacking, so fill here:
        df["ticker"] = req.tickers[0]

    # Ensure ticker column exists (for single ticker path)
    if "ticker" not in df.columns:
        df["ticker"] = req.tickers[0]

    return df


def default_ohlcv_path(req: OHLCVRequest, out_dir: Path = RAW_DIR) -> Path:
    tickers_slug = "-".join([t.replace("^", "") for t in req.tickers])
    end = req.end or "latest"
    filename = f"ohlcv_{tickers_slug}_{req.start}_{end}_{req.interval}.parquet"
    return out_dir / filename


def save_ohlcv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # parquet is compact and fast; easy to read later
    df.to_parquet(path, index=False)


def load_ohlcv(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def upsert_cache(req: OHLCVRequest, out_path: Optional[Path] = None) -> Path:
    """
    Fetch and save to a deterministic cache path.
    Returns the output path.
    """
    out_path = out_path or default_ohlcv_path(req)
    df = fetch_ohlcv(req)
    save_ohlcv(df, out_path)
    return out_path
