"""Pull historical bars from a running MT5 terminal and cache to parquet.

Design notes:
- The MT5 terminal must be running and logged into a demo account. We do not
  pass credentials through the Python API; the terminal's own login is used.
- XAUUSD on different brokers uses different symbol suffixes. We probe a small
  list of likely names rather than hard-coding "XAUUSD".
- Bars are cached per (symbol, timeframe) as a single parquet file. Subsequent
  calls only fetch the gap between cache_end and `end`, then concatenate.
- copy_rates_range is preferred over copy_rates_from because it returns all
  bars in the window in one call (the server caps per-call counts otherwise).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from mcmc_cuda.config import RAW_DIR
from mcmc_cuda.data.timeframes import get as get_timeframe
from mcmc_cuda.data.timeframes import to_mt5_constant

XAUUSD_CANDIDATES = ("XAUUSD", "XAUUSDm", "XAUUSD.m", "XAUUSD.r", "GOLD", "GOLDm")


@dataclass
class MT5Connection:
    """Context manager around mt5.initialize / shutdown."""

    def __enter__(self):
        import MetaTrader5 as mt5

        if not mt5.initialize():
            raise RuntimeError(
                f"mt5.initialize() failed: {mt5.last_error()}. "
                "Is the MT5 terminal running and logged in?"
            )
        self._mt5 = mt5
        return mt5

    def __exit__(self, exc_type, exc, tb):
        self._mt5.shutdown()


def resolve_symbol(mt5, requested: str = "XAUUSD") -> str:
    """Return the actual broker symbol for gold, probing common suffixes."""
    info = mt5.symbol_info(requested)
    if info is not None:
        if not info.visible:
            mt5.symbol_select(requested, True)
        return requested

    for candidate in XAUUSD_CANDIDATES:
        info = mt5.symbol_info(candidate)
        if info is not None:
            if not info.visible:
                mt5.symbol_select(candidate, True)
            return candidate

    raise RuntimeError(
        f"Could not resolve {requested!r} on this broker. "
        f"Tried: {XAUUSD_CANDIDATES}. Check Market Watch in MT5."
    )


def _cache_path(symbol: str, timeframe: str) -> Path:
    safe = symbol.replace(".", "_")
    return RAW_DIR / f"{safe}_{timeframe}.parquet"


def _rates_to_df(rates) -> pd.DataFrame:
    """MT5 rates ndarray -> tz-aware UTC DataFrame indexed by time."""
    df = pd.DataFrame(rates)
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    keep = ["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
    return df[[c for c in keep if c in df.columns]]


def fetch_bars(
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Return bars for [start, end] in UTC. Uses parquet cache when available.

    If `end` is None, fetches up to the latest available bar.
    """
    end = end or datetime.now(timezone.utc)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    get_timeframe(timeframe)  # validates name
    cache = _cache_path(symbol, timeframe)

    cached: pd.DataFrame | None = None
    fetch_start = start
    if use_cache and cache.exists():
        cached = pd.read_parquet(cache)
        if not cached.empty:
            cache_end = cached.index.max().to_pydatetime()
            if cache_end >= end:
                return cached.loc[start:end]
            # only fetch the gap
            fetch_start = max(start, cache_end)

    with MT5Connection() as mt5:
        broker_symbol = resolve_symbol(mt5, symbol)
        tf = to_mt5_constant(timeframe)
        # Server can be slow to wake the symbol; tiny pause helps on cold start.
        time.sleep(0.05)
        rates = mt5.copy_rates_range(broker_symbol, tf, fetch_start, end)
        if rates is None:
            raise RuntimeError(f"copy_rates_range returned None: {mt5.last_error()}")

    fresh = _rates_to_df(rates)

    if cached is not None and not cached.empty:
        merged = pd.concat([cached, fresh])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    else:
        merged = fresh

    if use_cache and not merged.empty:
        cache.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(cache)

    return merged.loc[start:end]


def sanity_check() -> dict:
    """Quick connectivity probe — used by scripts/sanity_mt5.py."""
    with MT5Connection() as mt5:
        acct = mt5.account_info()
        symbol = resolve_symbol(mt5, "XAUUSD")
        info = mt5.symbol_info(symbol)
        return {
            "account_login": getattr(acct, "login", None),
            "account_server": getattr(acct, "server", None),
            "trade_mode": getattr(acct, "trade_mode", None),
            "is_demo": getattr(acct, "trade_mode", None) == mt5.ACCOUNT_TRADE_MODE_DEMO,
            "balance": getattr(acct, "balance", None),
            "currency": getattr(acct, "currency", None),
            "resolved_symbol": symbol,
            "spread": getattr(info, "spread", None),
            "digits": getattr(info, "digits", None),
            "point": getattr(info, "point", None),
        }
