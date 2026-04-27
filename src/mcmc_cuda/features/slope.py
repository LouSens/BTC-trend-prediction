"""Rolling linear-regression slope of log-price.

Vectorized closed form (no pandas.rolling.apply): for window w with x=0..w-1,

    slope = Σ(x - x̄)(y - ȳ) / Σ(x - x̄)²

The denominator is constant in w (call it D = w(w-1)(w+1)/12). The numerator
expands to Σ x·y - w·x̄·ȳ, both terms computable via rolling sums.

Returns slope in *log-price units per bar* — a rate. Multiply by `bars_per_day`
for a daily-rate interpretation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_log_price_slope(close: pd.Series, window: int = 20) -> pd.Series:
    if window < 3:
        raise ValueError("window must be >= 3")
    y = np.log(close.astype(float))
    n = window
    x_bar = (n - 1) / 2.0
    denom = n * (n * n - 1) / 12.0  # Σ(x - x̄)² for x=0..n-1

    # Σ x·y over rolling window: use convolution of y with [0,1,2,...,n-1].
    weights = np.arange(n, dtype=np.float64)
    sum_xy = y.rolling(window).apply(lambda v: float(np.dot(weights, v)), raw=True)
    sum_y = y.rolling(window).sum()
    numer = sum_xy - x_bar * sum_y
    return (numer / denom).rename("slope")
