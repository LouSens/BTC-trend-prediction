"""Unit tests for the filter strategy layer."""
from __future__ import annotations

import numpy as np
import pandas as pd

from mcmc_cuda.strategy.filters import FilterConfig, apply_filters, compute_filter_frame


def _ohlc(n: int = 500, drift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rets = rng.normal(drift, 0.001, size=n)
    close = 2000.0 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.0005, size=n)))
    low = close * (1 - np.abs(rng.normal(0, 0.0005, size=n)))
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close}, index=idx
    )


def test_no_filters_passthrough():
    df = _ohlc()
    raw = pd.Series(1, index=df.index)
    cfg = FilterConfig()
    ff = compute_filter_frame(df["high"], df["low"], df["close"], cfg)
    out = apply_filters(raw, ff if not ff.empty else pd.DataFrame(index=df.index), cfg)
    assert (out == 1).all()


def test_slope_filter_kills_counter_trend():
    # Strong uptrend; short signals should be filtered out by slope.
    df = _ohlc(400, drift=+0.001)
    raw = pd.Series(-1, index=df.index)
    cfg = FilterConfig(use_slope=True, slope_window=20)
    ff = compute_filter_frame(df["high"], df["low"], df["close"], cfg)
    out = apply_filters(raw, ff, cfg)
    # most short signals should be zeroed since slope is positive
    n_kept = int((out == -1).sum())
    n_total = int(out.dropna().shape[0])
    assert n_kept / n_total < 0.5


def test_strength_filter_drops_weak_trend():
    df = _ohlc(400, drift=0.0)
    raw = pd.Series(1, index=df.index)
    cfg = FilterConfig(use_strength=True, adx_min=40.0)  # high bar
    ff = compute_filter_frame(df["high"], df["low"], df["close"], cfg)
    out = apply_filters(raw, ff, cfg)
    # With a high ADX threshold on noise, most signals should be filtered.
    assert (out == 0).mean() > 0.5


def test_momentum_filter_direction_alignment():
    df = _ohlc(400, drift=0.0)
    raw = pd.Series(1, index=df.index)  # only longs
    cfg = FilterConfig(use_momentum=True, rsi_long_min=50.0)
    ff = compute_filter_frame(df["high"], df["low"], df["close"], cfg)
    out = apply_filters(raw, ff, cfg)
    # Surviving longs should occur only on bars where rsi_dir == 1.
    keep = (out == 1)
    assert (ff.loc[keep, "rsi_dir"] == 1).all()
