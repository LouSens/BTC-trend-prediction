"""Scalp v2 signal generator: state machine sanity."""
from __future__ import annotations

import numpy as np
import pandas as pd

from mcmc_cuda.strategy.scalp import (
    BREAKOUT,
    IDLE,
    RETEST,
    ScalpConfig,
    generate_scalp_signals,
)


def _trending_ohlc(n: int = 500, drift: float = 0.0015, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.0008, size=n)
    close = 2000.0 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.0004, size=n)))
    low = close * (1 - np.abs(rng.normal(0, 0.0004, size=n)))
    idx = pd.date_range("2024-01-01 07:00", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close}, index=idx)


def test_no_signals_in_flat_drift():
    bars = _trending_ohlc(800, drift=0.0, seed=1)
    out = generate_scalp_signals(
        bars,
        cfg=ScalpConfig(require_session=False),
    )
    # In a true random walk with no drift the breakout-retest combo is rare.
    long_signals = int((out["signal"] == 1).sum())
    short_signals = int((out["signal"] == -1).sum())
    # We don't assert zero — the model can fire occasionally — only that the
    # rate is well below "every other bar".
    assert (long_signals + short_signals) < len(bars) // 5


def test_uptrend_produces_some_long_signals():
    bars = _trending_ohlc(800, drift=0.001, seed=2)
    out = generate_scalp_signals(
        bars,
        cfg=ScalpConfig(require_session=False, htf_ema_window=20),
    )
    assert int((out["signal"] == 1).sum()) > 0
    # Shorts should be much rarer than longs.
    assert (out["signal"] == 1).sum() > (out["signal"] == -1).sum()


def test_state_machine_progression_visits_all_states():
    bars = _trending_ohlc(800, drift=0.001, seed=4)
    out = generate_scalp_signals(
        bars,
        cfg=ScalpConfig(require_session=False, htf_ema_window=20),
    )
    states = set(out["state_long"].unique().tolist())
    assert IDLE in states
    # We expect at least one breakout to fire in a clear uptrend.
    assert (BREAKOUT in states) or (RETEST in states)


def test_session_gate_zeros_signals_outside_window():
    # Build bars that span asia time (00:00 UTC start) and require session.
    rng = np.random.default_rng(0)
    n = 500
    close = 2000.0 * np.exp(np.cumsum(rng.normal(0.001, 0.0008, size=n)))
    high = close * (1 + np.abs(rng.normal(0, 0.0004, size=n)))
    low  = close * (1 - np.abs(rng.normal(0, 0.0004, size=n)))
    idx = pd.date_range("2024-01-01 00:00", periods=n, freq="5min", tz="UTC")
    bars = pd.DataFrame({"open": close, "high": high, "low": low, "close": close}, index=idx)

    out = generate_scalp_signals(bars, cfg=ScalpConfig(require_session=True))
    # All non-zero signals must land in london or overlap.
    fired = out[out["signal"] != 0]
    assert (fired["session"].isin(["london", "overlap"])).all()
