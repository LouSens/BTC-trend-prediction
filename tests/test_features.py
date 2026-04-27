"""Unit tests for Phase 2 feature modules."""
from __future__ import annotations

import numpy as np
import pandas as pd

from mcmc_cuda.features.momentum import roc, rsi
from mcmc_cuda.features.slope import rolling_log_price_slope
from mcmc_cuda.features.strength import adx, atr, true_range


def _ohlc(n: int = 500, drift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rets = rng.normal(drift, 0.001, size=n)
    close = 2000.0 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.0005, size=n)))
    low = close * (1 - np.abs(rng.normal(0, 0.0005, size=n)))
    open_ = close.copy()
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close}, index=idx
    )


def test_slope_sign_matches_drift():
    up = _ohlc(300, drift=+0.001)
    dn = _ohlc(300, drift=-0.001)
    s_up = rolling_log_price_slope(up["close"], window=30).dropna()
    s_dn = rolling_log_price_slope(dn["close"], window=30).dropna()
    assert s_up.mean() > 0
    assert s_dn.mean() < 0


def test_rsi_in_range():
    df = _ohlc(500)
    r = rsi(df["close"], length=14)
    assert r.between(0, 100).all()


def test_roc_zero_for_constant():
    idx = pd.date_range("2024-01-01", periods=50, freq="15min", tz="UTC")
    s = pd.Series(2000.0, index=idx)
    r = roc(s, length=10).dropna()
    assert (r == 0).all()


def test_atr_positive():
    df = _ohlc(200)
    a = atr(df["high"], df["low"], df["close"], length=14).dropna()
    assert (a > 0).all()


def test_adx_columns_and_di_sum_under_drift():
    df = _ohlc(400, drift=+0.001)
    a = adx(df["high"], df["low"], df["close"], length=14)
    assert set(a.columns) == {"plus_di", "minus_di", "adx"}
    # Under positive drift, mean +DI should exceed mean -DI on average.
    tail = a.iloc[100:]
    assert tail["plus_di"].mean() > tail["minus_di"].mean()


def test_true_range_nonneg():
    df = _ohlc(100)
    tr = true_range(df["high"], df["low"], df["close"]).dropna()
    assert (tr >= 0).all()
