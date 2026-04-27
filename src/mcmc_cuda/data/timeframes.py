"""Timeframe string <-> MT5 constant mapping, isolated from the broker module.

Keeping this in `data/` (not `broker/`) so non-broker code can reason about
timeframes without importing MetaTrader5 (which is Windows-only and would
fail to import on CI / non-Windows machines).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Timeframe:
    name: str
    minutes: int


TIMEFRAMES: dict[str, Timeframe] = {
    "M1":  Timeframe("M1",  1),
    "M5":  Timeframe("M5",  5),
    "M15": Timeframe("M15", 15),
    "M30": Timeframe("M30", 30),
    "H1":  Timeframe("H1",  60),
    "H4":  Timeframe("H4",  240),
    "D1":  Timeframe("D1",  1440),
}


def get(name: str) -> Timeframe:
    if name not in TIMEFRAMES:
        raise KeyError(f"Unknown timeframe {name!r}. Known: {sorted(TIMEFRAMES)}")
    return TIMEFRAMES[name]


def to_mt5_constant(name: str):
    """Resolve to the MetaTrader5 module's TIMEFRAME_* constant, lazily."""
    import MetaTrader5 as mt5  # local import: Windows + terminal required

    return getattr(mt5, f"TIMEFRAME_{get(name).name}")
