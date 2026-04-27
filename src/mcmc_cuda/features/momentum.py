"""Momentum features: rate-of-change and RSI (Wilder)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def roc(close: pd.Series, length: int = 14) -> pd.Series:
    """Rate of Change as a fraction (0.01 = 1%)."""
    return (close / close.shift(length) - 1.0).rename("roc")


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Wilder's RSI in [0, 100]. Uses an EMA with alpha=1/length."""
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    alpha = 1.0 / length
    avg_up = up.ewm(alpha=alpha, adjust=False).mean()
    avg_dn = down.ewm(alpha=alpha, adjust=False).mean()
    rs = avg_up / avg_dn.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(50.0).rename("rsi")
