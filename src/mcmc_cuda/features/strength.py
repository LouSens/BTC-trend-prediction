"""Trend strength: ATR + ADX (Wilder).

ADX > 25 is the conventional "trending" cutoff; > 40 is strong trend.
We expose ADX, +DI, -DI, and ATR so downstream code can build either
a strength gate or a directional confirmation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(),
         (high - prev_close).abs(),
         (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rename("tr")


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Wilder's ATR — RMA (alpha=1/length) of True Range."""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / length, adjust=False).mean().rename("atr")


def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14
) -> pd.DataFrame:
    """Wilder's ADX with +DI / -DI components.

    Returns a DataFrame with columns: plus_di, minus_di, adx.
    """
    up = high.diff()
    down = -low.diff()

    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)

    tr = true_range(high, low, close)
    alpha = 1.0 / length
    atr_s = tr.ewm(alpha=alpha, adjust=False).mean()

    plus_di = 100.0 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_s.replace(0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_s.replace(0, np.nan)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_s = dx.ewm(alpha=alpha, adjust=False).mean()

    return pd.DataFrame(
        {"plus_di": plus_di.fillna(0), "minus_di": minus_di.fillna(0), "adx": adx_s.fillna(0)}
    )
