"""Microstructure feature tests: volume spike, sweep, OB, FVG."""
from __future__ import annotations

import numpy as np
import pandas as pd

from mcmc_cuda.features.microstructure import (
    detect_order_blocks,
    fair_value_gap,
    liquidity_sweep_long,
    liquidity_sweep_short,
    volume_spike,
    volume_zscore,
)


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-06-03 08:00", periods=n, freq="5min", tz="UTC")


# ----------------------------- volume spike -----------------------------
def test_volume_zscore_excludes_current_bar():
    v = pd.Series([1] * 20 + [10], index=_idx(21))
    z = volume_zscore(v, length=20)
    # The spike bar's z must be very large (mean=1, sd=0 -> nan -> we shifted).
    # With a constant prior series sd=0, z is NaN; so set up something with std.
    v2 = pd.Series(np.r_[np.ones(19), 2.0, 8.0], index=_idx(21))
    z2 = volume_zscore(v2, length=20)
    assert np.isfinite(z2.iloc[-1])
    assert z2.iloc[-1] > 2.0


def test_volume_spike_threshold():
    v = pd.Series(np.r_[np.ones(19) * 1.0, 1.5, 6.0], index=_idx(21))
    spk = volume_spike(v, length=20, z_threshold=2.0)
    assert bool(spk.iloc[-1]) is True
    # Earlier bars are below threshold.
    assert int(spk.iloc[:-1].sum()) == 0


# ----------------------------- liquidity sweeps -----------------------------
def test_liquidity_sweep_long_pierces_then_reclaims():
    n = 30
    high = pd.Series(np.full(n, 2010.0), index=_idx(n))
    low  = pd.Series(np.full(n, 1990.0), index=_idx(n))
    close = pd.Series(np.full(n, 2000.0), index=_idx(n))
    # Bar 25: low pierces below the 1990 swing low, close back above.
    low.iloc[25] = 1985.0
    close.iloc[25] = 1995.0
    out = liquidity_sweep_long(high, low, close, lookback=20)
    assert bool(out.iloc[25]) is True
    # No earlier sweep.
    assert int(out.iloc[:25].sum()) == 0


def test_liquidity_sweep_short_symmetric():
    n = 30
    high = pd.Series(np.full(n, 2010.0), index=_idx(n))
    low  = pd.Series(np.full(n, 1990.0), index=_idx(n))
    close = pd.Series(np.full(n, 2000.0), index=_idx(n))
    high.iloc[25] = 2015.0
    close.iloc[25] = 2005.0
    out = liquidity_sweep_short(high, low, close, lookback=20)
    assert bool(out.iloc[25]) is True


def test_liquidity_sweep_requires_close_back_through():
    n = 30
    high = pd.Series(np.full(n, 2010.0), index=_idx(n))
    low  = pd.Series(np.full(n, 1990.0), index=_idx(n))
    close = pd.Series(np.full(n, 2000.0), index=_idx(n))
    # Pierce but DON'T reclaim — close stays below the swing low.
    low.iloc[25] = 1985.0
    close.iloc[25] = 1987.0
    out = liquidity_sweep_long(high, low, close, lookback=20)
    assert bool(out.iloc[25]) is False


# ----------------------------- order blocks -----------------------------
def test_bullish_order_block_marks_last_bearish_before_impulse():
    # Build a tiny series: bar 14 is a bearish bar, bar 15 is a strong bullish
    # impulse > 1.5 * ATR -> bar 14 should be marked as bullish OB.
    n = 30
    rng = np.random.default_rng(0)
    base = 2000.0 + rng.normal(0, 0.5, size=n).cumsum()
    open_ = base.copy()
    close = base.copy()
    high = base + 1.0
    low  = base - 1.0
    # Bar 14: bearish (close < open).
    open_[14] = base[14] + 1.0
    close[14] = base[14] - 1.0
    high[14] = open_[14] + 0.5
    low[14]  = close[14] - 0.5
    # Bar 15: large bullish impulse.
    open_[15] = close[14]
    close[15] = open_[15] + 8.0     # big body
    high[15] = close[15] + 0.5
    low[15]  = open_[15] - 0.5
    idx = _idx(n)
    obs = detect_order_blocks(
        pd.Series(open_, idx), pd.Series(high, idx),
        pd.Series(low, idx), pd.Series(close, idx),
        impulse_atr_mult=1.5, atr_length=14,
    )
    assert np.isfinite(obs["bull_ob_high"].iloc[15])
    assert obs["bull_ob_high"].iloc[15] == high[14]
    assert obs["bull_ob_low"].iloc[15] == low[14]


# ----------------------------- FVG -----------------------------
def test_fair_value_gap_bullish():
    n = 5
    high = pd.Series([2000, 2001, 2002, 2003, 2004], index=_idx(n), dtype=float)
    low  = pd.Series([1995, 1996, 1997, 2002, 1999], index=_idx(n), dtype=float)
    close = pd.Series([1998, 1999, 2000, 2002.5, 2001], index=_idx(n), dtype=float)
    # At i=3: low[3]=2002 > high[1]=2001 -> bullish FVG.
    out = fair_value_gap(high, low, close)
    assert int(out["fvg_dir"].iloc[3]) == 1
    assert out["fvg_low"].iloc[3] == 2001.0   # h2
    assert out["fvg_high"].iloc[3] == 2002.0  # low[i]


def test_fair_value_gap_bearish():
    n = 5
    high = pd.Series([2010, 2009, 2008, 2000, 2002], index=_idx(n), dtype=float)
    low  = pd.Series([2005, 2004, 2003, 1995, 1996], index=_idx(n), dtype=float)
    close = pd.Series([2008, 2006, 2004, 1996, 1998], index=_idx(n), dtype=float)
    # At i=3: high[3]=2000 < low[1]=2004 -> bearish FVG.
    out = fair_value_gap(high, low, close)
    assert int(out["fvg_dir"].iloc[3]) == -1
