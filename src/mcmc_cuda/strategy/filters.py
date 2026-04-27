"""Filter layer that masks raw ensemble signals using regime/strength/slope/momentum.

A signal survives a filter iff the filter's directional vote agrees with the
signal sign (or the filter is non-directional and just gates on/off).

Each filter is *optional* via FilterConfig flags so we can isolate which one
moves the needle and which is dead weight.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mcmc_cuda.features.momentum import roc, rsi
from mcmc_cuda.features.regime import classify as classify_regime
from mcmc_cuda.features.regime import fit_regime
from mcmc_cuda.features.slope import rolling_log_price_slope
from mcmc_cuda.features.strength import adx


@dataclass
class FilterConfig:
    use_regime: bool = False
    use_strength: bool = False
    use_slope: bool = False
    use_momentum: bool = False

    # Regime: gate-only, requires "trending" state.
    regime_n_states: int = 2

    # Strength: ADX > adx_min AND DI alignment with signal direction.
    adx_length: int = 14
    adx_min: float = 25.0
    require_di_alignment: bool = True

    # Slope: 20-bar log-price slope sign must match signal.
    slope_window: int = 20

    # Momentum: RSI(length) > rsi_long_min for long, < rsi_short_max for short.
    rsi_length: int = 14
    rsi_long_min: float = 50.0
    rsi_short_max: float = 50.0


def compute_filter_frame(
    high: pd.Series, low: pd.Series, close: pd.Series, cfg: FilterConfig
) -> pd.DataFrame:
    """Return a DataFrame with one column per active filter giving its vote.

    Vote conventions per column:
      regime    : 1 if trending, 0 if ranging              (gate-only)
      adx_pass  : 1 if ADX >= threshold                    (gate-only)
      di_dir    : +1 if +DI > -DI, -1 otherwise            (directional)
      slope_dir : sign of slope                            (directional)
      rsi_dir   : +1 if rsi > rsi_long_min, -1 if rsi <    (directional)
                  rsi_short_max, else 0
    """
    out = pd.DataFrame(index=close.index)

    if cfg.use_regime:
        model = fit_regime(close, n_states=cfg.regime_n_states)
        out["regime"] = classify_regime(close, model)

    if cfg.use_strength:
        df = adx(high, low, close, length=cfg.adx_length)
        out["adx_pass"] = (df["adx"] >= cfg.adx_min).astype(np.int8)
        out["di_dir"] = np.where(df["plus_di"] > df["minus_di"], 1, -1).astype(np.int8)

    if cfg.use_slope:
        s = rolling_log_price_slope(close, window=cfg.slope_window)
        out["slope_dir"] = np.sign(s).fillna(0).astype(np.int8)

    if cfg.use_momentum:
        r = rsi(close, length=cfg.rsi_length)
        rsi_dir = pd.Series(0, index=close.index, dtype=np.int8)
        rsi_dir[r > cfg.rsi_long_min] = 1
        rsi_dir[r < cfg.rsi_short_max] = -1
        out["rsi_dir"] = rsi_dir
        out["rsi"] = r  # for diagnostics

    return out


def apply_filters(
    raw_signal: pd.Series, filter_frame: pd.DataFrame, cfg: FilterConfig
) -> pd.Series:
    """Combine raw signal and filter frame -> filtered signal.

    Rule: a non-zero raw signal is preserved iff every active filter agrees:
      - gate filters (regime, adx_pass): require value == 1
      - directional filters (di_dir, slope_dir, rsi_dir): require sign == raw signal sign
    """
    sig = raw_signal.reindex(filter_frame.index).fillna(0).astype(int).copy()
    sign = np.sign(sig)

    if cfg.use_regime and "regime" in filter_frame:
        sig = sig.where(filter_frame["regime"] == 1, 0)

    if cfg.use_strength:
        if "adx_pass" in filter_frame:
            sig = sig.where(filter_frame["adx_pass"] == 1, 0)
        if cfg.require_di_alignment and "di_dir" in filter_frame:
            sig = sig.where(filter_frame["di_dir"] == sign, 0)

    if cfg.use_slope and "slope_dir" in filter_frame:
        sig = sig.where(filter_frame["slope_dir"] == sign, 0)

    if cfg.use_momentum and "rsi_dir" in filter_frame:
        sig = sig.where(filter_frame["rsi_dir"] == sign, 0)

    return sig.astype(int).rename("signal")
