"""Scalp v2 — M15 bias + M5 breakout-retest momentum strategy.

State machine (per side, run independently for longs and shorts):

    IDLE
      └── HTF bias agrees + price prints a fresh `swing_lookback`-bar high
          (low for shorts) within the last `breakout_window` bars
          → BREAKOUT (record breakout level)

    BREAKOUT
      └── price pulls back at least `retest_atr_pct` × ATR below the
          breakout level (above for shorts) within `retest_window` bars
          → RETEST (record retest low/high)

    RETEST
      └── price closes back above the breakout level (below for shorts)
          AND short-term momentum agrees (close > prev close + RSI on side)
          → emit signal (+1 / -1) and reset to IDLE

If any leg of the chain expires (no breakout / no retest within window),
the side resets to IDLE.

This is intentionally rule-based and cheap — the goal is a robust scaffold
that survives live, not an over-fit signal.

Inputs:
- `bars` is the *execution* timeframe (M5 by default). It must contain
  open/high/low/close.
- `htf_close` is the *bias* timeframe close (M15 by default), reindexed to
  the bars timeframe via forward-fill. If None, defaults to using bars.close
  with an EMA-slope-based bias (still useful when only one timeframe is
  available).

Outputs a DataFrame indexed like `bars` with columns
    signal (-1/0/+1), htf_bias, breakout_level, retest_extreme, state_long,
    state_short, atr.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mcmc_cuda.features.momentum import rsi
from mcmc_cuda.features.strength import atr
from mcmc_cuda.strategy.sessions import (
    PREFERRED_SCALP_SESSIONS,
    label_sessions,
)


IDLE, BREAKOUT, RETEST = 0, 1, 2


@dataclass
class ScalpConfig:
    swing_lookback: int = 20            # bars for fresh-high/low test
    breakout_window: int = 10           # bars to allow breakout signal to live
    retest_window: int = 8              # bars to wait for the retest
    retest_atr_pct: float = 0.5         # min pullback in ATR units
    atr_length: int = 14
    htf_ema_window: int = 50            # used only when htf_close is None
    rsi_length: int = 14
    rsi_long_min: float = 50.0
    rsi_short_max: float = 50.0
    require_session: bool = True
    allowed_sessions: tuple[str, ...] = PREFERRED_SCALP_SESSIONS
    require_close_above_breakout: bool = True
    require_close_break: bool = True       # breakout requires a CLOSE above swing, not a wick
    signal_cooldown_bars: int = 6          # min bars between successive signals on the same side


def _htf_bias_series(close: pd.Series, htf_close: pd.Series | None, cfg: ScalpConfig) -> pd.Series:
    """+1/0/-1 directional bias from the HTF series, aligned to `close.index`."""
    if htf_close is None:
        ema = close.ewm(span=cfg.htf_ema_window, adjust=False).mean()
        slope = ema.diff(cfg.htf_ema_window)
        return np.sign(slope).fillna(0).astype(int)

    htf = htf_close.copy()
    if htf.index.tz is None and close.index.tz is not None:
        htf.index = htf.index.tz_localize(close.index.tz)
    ema = htf.ewm(span=cfg.htf_ema_window, adjust=False).mean()
    slope = np.sign(ema.diff(cfg.htf_ema_window)).fillna(0).astype(int)
    return slope.reindex(close.index, method="ffill").fillna(0).astype(int)


def generate_scalp_signals(
    bars: pd.DataFrame,
    htf_close: pd.Series | None = None,
    cfg: ScalpConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or ScalpConfig()
    o, h, l, c = bars["open"], bars["high"], bars["low"], bars["close"]
    n = len(bars)

    a = atr(h, l, c, length=cfg.atr_length).values
    r = rsi(c, length=cfg.rsi_length).values
    htf_bias = _htf_bias_series(c, htf_close, cfg).values

    # Rolling swings: highest high / lowest low over previous `swing_lookback`
    # bars (excluding current). Computed by shifting before rolling.
    rolling_high = h.rolling(cfg.swing_lookback).max().shift(1).values
    rolling_low  = l.rolling(cfg.swing_lookback).min().shift(1).values

    # Session mask.
    sessions = label_sessions(bars.index)
    if cfg.require_session:
        sess_ok = sessions.isin(cfg.allowed_sessions).values
    else:
        sess_ok = np.ones(n, dtype=bool)

    signal = np.zeros(n, dtype=np.int8)
    state_long  = np.zeros(n, dtype=np.int8)
    state_short = np.zeros(n, dtype=np.int8)
    breakout_level = np.full(n, np.nan)
    retest_extreme = np.full(n, np.nan)

    # Per-side state machine (long).
    sl_state, sl_breakout, sl_retest, sl_age = IDLE, np.nan, np.nan, 0
    long_cooldown = 0
    # Per-side state machine (short).
    ss_state, ss_breakout, ss_retest, ss_age = IDLE, np.nan, np.nan, 0
    short_cooldown = 0

    c_arr = c.values
    h_arr = h.values
    l_arr = l.values
    o_arr = o.values

    for i in range(n):
        atr_i = a[i] if np.isfinite(a[i]) else np.nan
        prev_c = c_arr[i - 1] if i > 0 else c_arr[i]

        if long_cooldown > 0:
            long_cooldown -= 1
        if short_cooldown > 0:
            short_cooldown -= 1

        # ---------------- LONG side ----------------
        if sl_state == IDLE:
            # Need fresh swing-high break with HTF bias up.
            broke_high = (
                c_arr[i] > rolling_high[i]
                if cfg.require_close_break
                else h_arr[i] > rolling_high[i]
            )
            if (
                htf_bias[i] > 0
                and np.isfinite(rolling_high[i])
                and broke_high
            ):
                sl_state = BREAKOUT
                sl_breakout = float(rolling_high[i])
                sl_age = 0
        elif sl_state == BREAKOUT:
            sl_age += 1
            # Check pullback to retest zone.
            if (
                np.isfinite(atr_i)
                and l_arr[i] <= sl_breakout - cfg.retest_atr_pct * atr_i
            ):
                sl_state = RETEST
                sl_retest = float(l_arr[i])
                sl_age = 0
            elif sl_age >= cfg.breakout_window:
                sl_state = IDLE
                sl_breakout = sl_retest = np.nan
        elif sl_state == RETEST:
            sl_age += 1
            # Trigger: close back above breakout level + momentum agree.
            momentum_ok = (
                c_arr[i] > prev_c
                and (not np.isfinite(r[i]) or r[i] >= cfg.rsi_long_min)
            )
            close_back = (
                c_arr[i] >= sl_breakout
                if cfg.require_close_above_breakout
                else c_arr[i] > sl_retest
            )
            if close_back and momentum_ok and sess_ok[i] and htf_bias[i] > 0 and long_cooldown == 0:
                signal[i] = 1
                long_cooldown = cfg.signal_cooldown_bars
                sl_state = IDLE
                sl_breakout = sl_retest = np.nan
            elif sl_age >= cfg.retest_window:
                sl_state = IDLE
                sl_breakout = sl_retest = np.nan

        # ---------------- SHORT side ----------------
        if ss_state == IDLE:
            broke_low = (
                c_arr[i] < rolling_low[i]
                if cfg.require_close_break
                else l_arr[i] < rolling_low[i]
            )
            if (
                htf_bias[i] < 0
                and np.isfinite(rolling_low[i])
                and broke_low
            ):
                ss_state = BREAKOUT
                ss_breakout = float(rolling_low[i])
                ss_age = 0
        elif ss_state == BREAKOUT:
            ss_age += 1
            if (
                np.isfinite(atr_i)
                and h_arr[i] >= ss_breakout + cfg.retest_atr_pct * atr_i
            ):
                ss_state = RETEST
                ss_retest = float(h_arr[i])
                ss_age = 0
            elif ss_age >= cfg.breakout_window:
                ss_state = IDLE
                ss_breakout = ss_retest = np.nan
        elif ss_state == RETEST:
            ss_age += 1
            momentum_ok = (
                c_arr[i] < prev_c
                and (not np.isfinite(r[i]) or r[i] <= cfg.rsi_short_max)
            )
            close_back = (
                c_arr[i] <= ss_breakout
                if cfg.require_close_above_breakout
                else c_arr[i] < ss_retest
            )
            if (close_back and momentum_ok and sess_ok[i]
                    and htf_bias[i] < 0 and short_cooldown == 0):
                if signal[i] == 0:           # don't fire both sides the same bar
                    signal[i] = -1
                    short_cooldown = cfg.signal_cooldown_bars
                ss_state = IDLE
                ss_breakout = ss_retest = np.nan
            elif ss_age >= cfg.retest_window:
                ss_state = IDLE
                ss_breakout = ss_retest = np.nan

        state_long[i]   = sl_state
        state_short[i]  = ss_state
        breakout_level[i] = sl_breakout if sl_state != IDLE else (
            ss_breakout if ss_state != IDLE else np.nan
        )
        retest_extreme[i] = sl_retest if sl_state == RETEST else (
            ss_retest if ss_state == RETEST else np.nan
        )

    return pd.DataFrame(
        {
            "signal": signal.astype(int),
            "htf_bias": htf_bias,
            "breakout_level": breakout_level,
            "retest_extreme": retest_extreme,
            "state_long": state_long,
            "state_short": state_short,
            "atr": a,
            "session": sessions.values,
        },
        index=bars.index,
    )


def resample_for_htf(bars: pd.DataFrame, htf: str = "15min") -> pd.DataFrame:
    """Helper to build an HTF OHLC frame from a finer one (e.g. M5 -> M15)."""
    out = pd.DataFrame({
        "open":  bars["open"].resample(htf).first(),
        "high":  bars["high"].resample(htf).max(),
        "low":   bars["low"].resample(htf).min(),
        "close": bars["close"].resample(htf).last(),
    }).dropna()
    return out
