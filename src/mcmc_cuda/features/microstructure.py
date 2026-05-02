"""Market microstructure features for SMC/ICT-style scalping.

We do NOT have a real L2 order book — only OHLC + MT5 tick_volume. So the
"order flow" features here are honest *proxies*, not the genuine article.
Specifically:

- Volume spike      : tick-volume z-score over a rolling window. Tick count
                      correlates with trader activity / institutional bursts
                      on XAUUSD reasonably well, but it is not order-flow
                      delta. Treat it as "unusual activity" detection.
- Liquidity sweep   : a candle that wicks beyond a recent swing extreme and
                      *closes back through it* — the classic stop-run pattern.
- Bullish/Bearish OB: the last opposite-direction candle before an impulsive
                      move. Used as a price *zone* (open..close range) where
                      institutions are presumed to have rested orders.
- Fair value gap    : a 3-bar imbalance — bar[i].low > bar[i-2].high (bullish)
                      or bar[i].high < bar[i-2].low (bearish). The gap is an
                      area price often revisits.

All features are computed using only past data; no look-ahead. Returned
Series are aligned to the input index.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mcmc_cuda.features.strength import atr


# ----------------------------------------------------------------------
# Volume spike
# ----------------------------------------------------------------------
def volume_zscore(volume: pd.Series, length: int = 20) -> pd.Series:
    """Rolling z-score of (tick) volume over `length` bars (excluding current).

    Excluding the current bar prevents look-ahead and forces the z-score to
    measure how unusual *this* bar's volume is vs. the recent past.
    """
    prior = volume.shift(1)
    mu = prior.rolling(length).mean()
    sd = prior.rolling(length).std(ddof=0)
    z = (volume - mu) / sd.replace(0, np.nan)
    return z.rename("vol_z")


def volume_spike(
    volume: pd.Series, length: int = 20, z_threshold: float = 2.0
) -> pd.Series:
    """Boolean Series: True when current bar's volume z-score >= threshold."""
    z = volume_zscore(volume, length=length)
    return (z >= z_threshold).fillna(False).rename("vol_spike")


# ----------------------------------------------------------------------
# Liquidity sweeps
# ----------------------------------------------------------------------
def liquidity_sweep_long(
    high: pd.Series, low: pd.Series, close: pd.Series,
    lookback: int = 20,
) -> pd.Series:
    """Bullish sweep: the bar's low pierces the lowest low of the previous
    `lookback` bars AND the bar closes back above that swing low.

    Interpretation: stops below a swing low were taken out and price recovered
    — trapped sellers / liquidity grab below support.
    """
    swing_low = low.shift(1).rolling(lookback).min()
    pierced = low < swing_low
    reclaimed = close > swing_low
    return (pierced & reclaimed).fillna(False).rename("sweep_long")


def liquidity_sweep_short(
    high: pd.Series, low: pd.Series, close: pd.Series,
    lookback: int = 20,
) -> pd.Series:
    """Bearish sweep: high pierces swing high; close back below it."""
    swing_high = high.shift(1).rolling(lookback).max()
    pierced = high > swing_high
    reclaimed = close < swing_high
    return (pierced & reclaimed).fillna(False).rename("sweep_short")


def sweep_extreme_long(
    low: pd.Series, lookback: int = 20
) -> pd.Series:
    """The pierced swing low for each bar — useful to place stops below it."""
    return low.shift(1).rolling(lookback).min().rename("sweep_low")


def sweep_extreme_short(
    high: pd.Series, lookback: int = 20
) -> pd.Series:
    return high.shift(1).rolling(lookback).max().rename("sweep_high")


# ----------------------------------------------------------------------
# Order blocks
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class OrderBlock:
    """A single order-block zone with a bar-index validity window."""
    side: int            # +1 bullish OB, -1 bearish OB
    formed_at: int       # bar index when the OB was identified
    high: float
    low: float


def detect_order_blocks(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
    impulse_atr_mult: float = 1.5,
    atr_length: int = 14,
) -> pd.DataFrame:
    """Per-bar OB markers.

    A *bullish* OB is the last bearish candle (close < open) before an
    impulsive bullish move where the next bar's close - open >= impulse_atr_mult * ATR.
    The OB's zone is [low_of_OB_bar .. high_of_OB_bar].

    Returned columns (NaN when no OB):
        bull_ob_high, bull_ob_low, bull_ob_age   (bars since last bullish OB)
        bear_ob_high, bear_ob_low, bear_ob_age
    """
    a = atr(high, low, close, length=atr_length)
    body = (close - open_)

    n = len(close)
    bull_high = np.full(n, np.nan)
    bull_low  = np.full(n, np.nan)
    bull_age  = np.full(n, np.nan)
    bear_high = np.full(n, np.nan)
    bear_low  = np.full(n, np.nan)
    bear_age  = np.full(n, np.nan)

    o = open_.values
    h = high.values
    lo = low.values
    c = close.values
    a_arr = a.values
    b = body.values

    last_bull: int | None = None
    last_bear: int | None = None
    last_bull_hl: tuple[float, float] | None = None
    last_bear_hl: tuple[float, float] | None = None

    for i in range(1, n):
        # Look at previous bar (i-1) — was it a candidate OB followed by an
        # impulse on bar i?
        impulse = b[i]
        atr_i = a_arr[i] if np.isfinite(a_arr[i]) else np.nan
        if np.isfinite(atr_i):
            # Bullish impulse on bar i; previous bar (i-1) must have been
            # bearish-bodied. Mark (i-1) as the bullish OB.
            if impulse >= impulse_atr_mult * atr_i and b[i - 1] < 0:
                last_bull = i - 1
                last_bull_hl = (float(h[i - 1]), float(lo[i - 1]))
            # Bearish impulse mirror.
            if impulse <= -impulse_atr_mult * atr_i and b[i - 1] > 0:
                last_bear = i - 1
                last_bear_hl = (float(h[i - 1]), float(lo[i - 1]))

        if last_bull is not None and last_bull_hl is not None:
            bull_high[i] = last_bull_hl[0]
            bull_low[i]  = last_bull_hl[1]
            bull_age[i]  = i - last_bull
        if last_bear is not None and last_bear_hl is not None:
            bear_high[i] = last_bear_hl[0]
            bear_low[i]  = last_bear_hl[1]
            bear_age[i]  = i - last_bear

    return pd.DataFrame(
        {
            "bull_ob_high": bull_high,
            "bull_ob_low":  bull_low,
            "bull_ob_age":  bull_age,
            "bear_ob_high": bear_high,
            "bear_ob_low":  bear_low,
            "bear_ob_age":  bear_age,
        },
        index=close.index,
    )


def in_zone(price: pd.Series, zone_low: pd.Series, zone_high: pd.Series) -> pd.Series:
    """Boolean: price within [zone_low, zone_high] inclusive."""
    ok = (price >= zone_low) & (price <= zone_high)
    return ok.fillna(False)


# ----------------------------------------------------------------------
# Fair value gap (3-bar imbalance)
# ----------------------------------------------------------------------
def fair_value_gap(
    high: pd.Series, low: pd.Series, close: pd.Series,
) -> pd.DataFrame:
    """A 3-bar FVG forms at bar i when:
      bullish FVG  : low[i] > high[i-2]   -> gap zone = (high[i-2], low[i])
      bearish FVG  : high[i] < low[i-2]   -> gap zone = (high[i], low[i-2])

    Returns per-bar columns (NaN when no FVG present *at this bar*; an FVG is
    only "present" on the bar it is identified — callers should track it
    forward themselves if they want a rolling list).
        fvg_dir     +1 / -1 / 0
        fvg_high    upper edge of the gap zone
        fvg_low     lower edge of the gap zone
    """
    h2 = high.shift(2)
    l2 = low.shift(2)
    bull = low > h2
    bear = high < l2

    fvg_dir = pd.Series(0, index=close.index, dtype=np.int8)
    fvg_dir[bull] = 1
    fvg_dir[bear] = -1

    fvg_high = pd.Series(np.nan, index=close.index)
    fvg_low  = pd.Series(np.nan, index=close.index)
    fvg_high[bull] = low[bull]
    fvg_low[bull]  = h2[bull]
    fvg_high[bear] = l2[bear]
    fvg_low[bear]  = high[bear]

    return pd.DataFrame(
        {"fvg_dir": fvg_dir, "fvg_high": fvg_high, "fvg_low": fvg_low}
    )
