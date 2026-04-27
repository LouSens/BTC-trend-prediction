"""Unified bar loader: prefer MT5, fall back to CSV/parquet on disk.

Lets you develop the GPU/backtest stack on a machine where MT5 isn't
installed yet, by pointing at a CSV dump of XAUUSD bars.

CSV format expected (header required):
    time,open,high,low,close,tick_volume[,spread,real_volume]

`time` is parsed as UTC. Either ISO-8601 strings or unix epoch seconds work.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from mcmc_cuda.config import RAW_DIR


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load bars from CSV/parquet. Returns UTC-indexed OHLCV DataFrame."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)

    if "time" in df.columns:
        # Accept both epoch seconds and ISO strings.
        if pd.api.types.is_numeric_dtype(df["time"]):
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        else:
            df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time")
    elif df.index.name in ("time", "Datetime", "Date", "datetime"):
        df.index = pd.to_datetime(df.index, utc=True)

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    return df


def load_bars(
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime | None = None,
    csv_path: str | Path | None = None,
    use_mt5: bool = True,
) -> pd.DataFrame:
    """Try MT5 first; fall back to CSV/parquet on failure or when use_mt5=False.

    If `csv_path` is None, falls back to `data/raw/<symbol>_<timeframe>.csv`
    or the parquet cache produced by the MT5 loader.
    """
    end = end or datetime.now(timezone.utc)

    if use_mt5:
        try:
            from mcmc_cuda.data.mt5_loader import fetch_bars

            return fetch_bars(symbol, timeframe, start, end)
        except Exception as e:
            if csv_path is None and not _default_csv(symbol, timeframe).exists():
                raise
            print(f"[loader] MT5 fetch failed ({e}); falling back to local file.")

    path = Path(csv_path) if csv_path else _default_csv(symbol, timeframe)
    if not path.exists():
        # also try the parquet cache
        pq = RAW_DIR / f"{symbol}_{timeframe}.parquet"
        if pq.exists():
            path = pq
        else:
            raise FileNotFoundError(
                f"No local bars found at {path} (and MT5 was disabled or unavailable)."
            )
    df = load_csv(path)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    return df.loc[start:end]


def _default_csv(symbol: str, timeframe: str) -> Path:
    return RAW_DIR / f"{symbol}_{timeframe}.csv"
