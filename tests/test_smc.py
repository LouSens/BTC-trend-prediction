"""SMC strategy: state machine + confluence gating."""
from __future__ import annotations

import numpy as np
import pandas as pd

from mcmc_cuda.strategy.smc import SMCConfig, generate_smc_signals


def _bars_with_long_sweep(n: int = 80) -> pd.DataFrame:
    """Engineered persistent uptrend with a sweep + reclaim around bar 50.

    Layout (so HTF EMA bias stays clearly UP at the sweep):
      bars 0..44  : monotonic uptrend 1990 -> 2030 (HTF bias clearly positive)
      bars 45..49 : tight consolidation forming swing low ~2025
      bar 50      : wicks to ~2018, closes 2027  -> sweep_long fires
      bar 50      : tick_volume z-score spike >> 2
      bars 51..55 : follow-through up
    """
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-06-03 08:00", periods=n, freq="15min", tz="UTC")

    open_ = np.zeros(n)
    close = np.zeros(n)
    high  = np.zeros(n)
    low   = np.zeros(n)

    # Phase 1: monotonic uptrend.
    for k in range(45):
        base = 1990.0 + (k / 44.0) * 40.0   # 1990 -> 2030
        open_[k] = base
        close[k] = base + 0.5
        high[k]  = base + 0.7
        low[k]   = base - 0.2

    # Phase 2: tight consolidation forming the swing low ~2025.
    for k in range(45, 50):
        open_[k] = 2027.0 + rng.normal(0, 0.1)
        close[k] = 2027.5 + rng.normal(0, 0.1)
        high[k]  = 2028.0 + rng.normal(0, 0.1)
        low[k]   = 2025.5 + rng.normal(0, 0.1)

    # Sweep bar 50: pierces the 20-bar rolling low (~2017) AND closes back
    # above it. We deliberately wick deeper so the test is robust to noise.
    open_[50] = 2027.0
    low[50]   = 2010.0
    high[50]  = 2028.0
    close[50] = 2027.0

    # Phase 3: follow-through.
    for k in range(51, 56):
        open_[k] = close[k - 1]
        close[k] = open_[k] + 1.5
        high[k]  = close[k] + 0.3
        low[k]   = open_[k] - 0.2
    for k in range(56, n):
        open_[k] = close[k - 1]
        close[k] = open_[k] + rng.normal(0, 0.2)
        high[k]  = max(open_[k], close[k]) + 0.3
        low[k]   = min(open_[k], close[k]) - 0.3

    # Volume jitter so the rolling std is non-zero; spike on bar 50.
    vol = np.full(n, 1000.0) + rng.normal(0, 80.0, size=n)
    vol[50] = 6000.0

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "tick_volume": vol},
        index=idx,
    )


def test_smc_emits_long_after_sweep_and_volume():
    bars = _bars_with_long_sweep()
    cfg = SMCConfig(
        sweep_lookback=20,
        arm_window=6,
        vol_spike_z=2.0,
        require_volume_spike=True,
        require_ob_or_fvg=False,         # don't require OB/FVG for this isolated test
        require_session=False,
        htf_ema_window=10,
    )
    out = generate_smc_signals(bars, cfg=cfg)
    # Some long signal must fire in bars 50..56.
    assert int((out["signal"].iloc[50:60] == 1).sum()) >= 1


def test_smc_no_volume_spike_blocks_signal():
    bars = _bars_with_long_sweep()
    # Flatten volume so the spike never fires.
    bars["tick_volume"] = 1000.0
    cfg = SMCConfig(
        require_volume_spike=True,
        require_ob_or_fvg=False,
        require_session=False,
        htf_ema_window=10,
    )
    out = generate_smc_signals(bars, cfg=cfg)
    assert int((out["signal"] != 0).sum()) == 0


def test_smc_falls_back_when_tick_volume_missing():
    bars = _bars_with_long_sweep().drop(columns=["tick_volume"])
    cfg = SMCConfig(
        require_volume_spike=False,    # caller already disabled it
        require_ob_or_fvg=False,
        require_session=False,
        htf_ema_window=10,
    )
    out = generate_smc_signals(bars, cfg=cfg)
    # Should still produce some signals via the sweep alone.
    assert int((out["signal"] != 0).sum()) >= 1


def test_smc_session_filter_blocks_asia():
    bars = _bars_with_long_sweep()
    # Shift the index into Asia (00:00-07:00 UTC).
    bars.index = pd.date_range("2024-06-03 00:00", periods=len(bars), freq="15min", tz="UTC")
    cfg = SMCConfig(
        require_volume_spike=True, require_ob_or_fvg=False,
        require_session=True, htf_ema_window=10,
    )
    out = generate_smc_signals(bars, cfg=cfg)
    fired = out[out["signal"] != 0]
    # All firings must lie inside the configured allowed sessions.
    assert (fired["session"].isin(["london", "overlap"])).all() or fired.empty
