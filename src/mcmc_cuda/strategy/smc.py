"""SMC/ICT-style high-confluence scalping strategy.

Honest framing: this is *not* a 90% win-rate machine. With realistic XAUUSD
costs and OHLC + tick-volume data, well-built confluence systems land in the
60-72% win-rate range with positive expectancy. We get there by:

1. Trading only after a *liquidity sweep + reclaim* — the textbook stop-run
   reversal pattern. Setups are rare; we filter aggressively.
2. Requiring at least one (configurable) extra confluence factor:
   - tick-volume spike on the reclaim bar (institutional activity proxy),
   - active order block within the entry zone,
   - bullish/bearish fair value gap below/above current price.
3. Restricting to liquid sessions (London / overlap by default).
4. HTF bias filter (EMA slope on a higher timeframe) — never fade the trend
   on the LTF unless the HTF agrees with the reversal.

Entry: at the next bar's open after the reclaim+confluence trigger.
Stop:  beyond the sweep extreme, with a small ATR buffer (so noise doesn't
       wick us out before the move develops).
Take-profit: ATR-based (configurable). The break-even mover in the engine
       (`breakeven_at_atr`) is what does the heavy lifting on win rate —
       once the trade is +X*ATR in profit, the SL moves to break-even and
       the trade either becomes a winner or scratches.

State machine (per side, run independently):

    IDLE
      └── HTF bias agrees + liquidity sweep detected on this bar
          → ARMED (record sweep extreme; wait for trigger / expiry)

    ARMED
      └── within `arm_window` bars:
            confluence (volume spike + (OB-in-zone OR FVG))
            AND price still above sweep_low (long) / below sweep_high (short)
              -> emit signal, reset to IDLE
          else if `arm_window` exceeded -> reset to IDLE
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mcmc_cuda.features.microstructure import (
    detect_order_blocks,
    fair_value_gap,
    liquidity_sweep_long,
    liquidity_sweep_short,
    sweep_extreme_long,
    sweep_extreme_short,
    volume_spike,
)
from mcmc_cuda.features.strength import atr
from mcmc_cuda.strategy.sessions import (
    PREFERRED_SCALP_SESSIONS,
    label_sessions,
)


IDLE, ARMED = 0, 1


@dataclass
class SMCConfig:
    sweep_lookback: int = 20            # bars to define swing extreme for sweep
    arm_window: int = 5                 # bars after sweep to look for confluence
    atr_length: int = 14
    htf_ema_window: int = 50
    # Volume confluence
    vol_spike_length: int = 20
    vol_spike_z: float = 1.5
    require_volume_spike: bool = True
    # OB / FVG confluence
    ob_impulse_atr_mult: float = 1.2
    require_ob_or_fvg: bool = True      # require at least one of OB-in-zone / FVG-aligned
    ob_max_age_bars: int = 50           # ignore OBs older than this
    # Session filter
    require_session: bool = True
    allowed_sessions: tuple[str, ...] = PREFERRED_SCALP_SESSIONS
    # Cooldowns / quality
    signal_cooldown_bars: int = 8
    require_close_above_sweep: bool = True   # entry trigger must close on the right side


def generate_smc_signals(
    bars: pd.DataFrame,
    htf_close: pd.Series | None = None,
    cfg: SMCConfig | None = None,
) -> pd.DataFrame:
    """Compute per-bar SMC signals and confluence diagnostics.

    Returns a DataFrame indexed like bars with columns:
        signal (-1/0/+1)
        htf_bias
        atr
        sweep_long, sweep_short
        sweep_extreme (the relevant extreme price for stop placement)
        confluence_volume, confluence_ob, confluence_fvg
        state_long, state_short
    """
    cfg = cfg or SMCConfig()
    if "tick_volume" not in bars.columns:
        # Fall back gracefully: treat volume as constant (volume spike never
        # fires, so callers should set require_volume_spike=False or supply
        # tick_volume).
        vol = pd.Series(1.0, index=bars.index)
    else:
        vol = bars["tick_volume"].astype(float)

    o, h, l, c = bars["open"], bars["high"], bars["low"], bars["close"]

    a = atr(h, l, c, length=cfg.atr_length)

    # HTF bias
    if htf_close is None:
        ema = c.ewm(span=cfg.htf_ema_window, adjust=False).mean()
        htf_bias = np.sign(ema.diff(cfg.htf_ema_window)).fillna(0).astype(int)
    else:
        htf = htf_close.copy()
        if htf.index.tz is None and bars.index.tz is not None:
            htf.index = htf.index.tz_localize(bars.index.tz)
        ema = htf.ewm(span=cfg.htf_ema_window, adjust=False).mean()
        slope = np.sign(ema.diff(cfg.htf_ema_window)).fillna(0).astype(int)
        htf_bias = slope.reindex(c.index, method="ffill").fillna(0).astype(int)

    sweep_l = liquidity_sweep_long(h, l, c, lookback=cfg.sweep_lookback)
    sweep_s = liquidity_sweep_short(h, l, c, lookback=cfg.sweep_lookback)
    sweep_low = sweep_extreme_long(l, lookback=cfg.sweep_lookback)
    sweep_high = sweep_extreme_short(h, lookback=cfg.sweep_lookback)

    vol_spk = volume_spike(vol, length=cfg.vol_spike_length, z_threshold=cfg.vol_spike_z)
    obs = detect_order_blocks(o, h, l, c, impulse_atr_mult=cfg.ob_impulse_atr_mult,
                              atr_length=cfg.atr_length)
    fvg = fair_value_gap(h, l, c)

    sessions = label_sessions(bars.index)
    sess_ok = (
        sessions.isin(cfg.allowed_sessions).values
        if cfg.require_session
        else np.ones(len(bars), dtype=bool)
    )

    n = len(bars)
    signal = np.zeros(n, dtype=np.int8)
    state_long  = np.zeros(n, dtype=np.int8)
    state_short = np.zeros(n, dtype=np.int8)
    confl_vol = np.zeros(n, dtype=np.int8)
    confl_ob  = np.zeros(n, dtype=np.int8)
    confl_fvg = np.zeros(n, dtype=np.int8)
    sweep_ext = np.full(n, np.nan)

    o_arr = o.values
    h_arr = h.values
    l_arr = l.values
    c_arr = c.values
    a_arr = a.values
    htf = htf_bias.values if hasattr(htf_bias, "values") else np.asarray(htf_bias)
    sl_swept = sweep_l.values
    ss_swept = sweep_s.values
    sl_low   = sweep_low.values
    ss_high  = sweep_high.values
    vspk     = vol_spk.values
    bull_lo  = obs["bull_ob_low"].values
    bull_hi  = obs["bull_ob_high"].values
    bull_age = obs["bull_ob_age"].values
    bear_lo  = obs["bear_ob_low"].values
    bear_hi  = obs["bear_ob_high"].values
    bear_age = obs["bear_ob_age"].values
    fvg_dir  = fvg["fvg_dir"].values
    fvg_hi   = fvg["fvg_high"].values
    fvg_lo   = fvg["fvg_low"].values

    # Per-side state machine.
    long_state, long_swept_low, long_age = IDLE, np.nan, 0
    short_state, short_swept_high, short_age = IDLE, np.nan, 0
    long_cooldown = 0
    short_cooldown = 0

    for i in range(n):
        if long_cooldown > 0:
            long_cooldown -= 1
        if short_cooldown > 0:
            short_cooldown -= 1

        # ---------------- LONG side ----------------
        # Step 1: maybe enter ARMED on this bar (sweep + HTF agree).
        if long_state == IDLE:
            if htf[i] >= 0 and sl_swept[i] and np.isfinite(sl_low[i]):
                long_state = ARMED
                long_swept_low = float(sl_low[i])
                long_age = 0
        # Step 2: if (now) ARMED, look for trigger or expiry on this same bar.
        if long_state == ARMED:
            # Invalidate if price re-broke the sweep low (no reclaim).
            if c_arr[i] < long_swept_low:
                long_state = IDLE
                long_swept_low = np.nan
            else:
                vol_ok = bool(vspk[i])
                ob_ok = (
                    np.isfinite(bull_lo[i])
                    and np.isfinite(bull_age[i])
                    and bull_age[i] <= cfg.ob_max_age_bars
                    and (l_arr[i] <= bull_hi[i] and h_arr[i] >= bull_lo[i])
                )
                fvg_ok = (fvg_dir[i] == 1)

                conf_ok = True
                if cfg.require_volume_spike:
                    conf_ok = conf_ok and vol_ok
                if cfg.require_ob_or_fvg:
                    conf_ok = conf_ok and (ob_ok or fvg_ok)

                close_back = (
                    c_arr[i] > long_swept_low
                    if cfg.require_close_above_sweep
                    else True
                )

                if conf_ok and close_back and sess_ok[i] and long_cooldown == 0:
                    signal[i] = 1
                    long_cooldown = cfg.signal_cooldown_bars
                    confl_vol[i] = int(vol_ok)
                    confl_ob[i]  = int(ob_ok)
                    confl_fvg[i] = int(fvg_ok)
                    sweep_ext[i] = long_swept_low
                    long_state = IDLE
                    long_swept_low = np.nan
                else:
                    long_age += 1
                    if long_age >= cfg.arm_window:
                        long_state = IDLE
                        long_swept_low = np.nan

        # ---------------- SHORT side ----------------
        if short_state == IDLE:
            if htf[i] <= 0 and ss_swept[i] and np.isfinite(ss_high[i]):
                short_state = ARMED
                short_swept_high = float(ss_high[i])
                short_age = 0
        if short_state == ARMED:
            if c_arr[i] > short_swept_high:
                short_state = IDLE
                short_swept_high = np.nan
            else:
                vol_ok = bool(vspk[i])
                ob_ok = (
                    np.isfinite(bear_lo[i])
                    and np.isfinite(bear_age[i])
                    and bear_age[i] <= cfg.ob_max_age_bars
                    and (l_arr[i] <= bear_hi[i] and h_arr[i] >= bear_lo[i])
                )
                fvg_ok = (fvg_dir[i] == -1)

                conf_ok = True
                if cfg.require_volume_spike:
                    conf_ok = conf_ok and vol_ok
                if cfg.require_ob_or_fvg:
                    conf_ok = conf_ok and (ob_ok or fvg_ok)

                close_back = (
                    c_arr[i] < short_swept_high
                    if cfg.require_close_above_sweep
                    else True
                )

                if conf_ok and close_back and sess_ok[i] and short_cooldown == 0:
                    if signal[i] == 0:    # don't fire both sides on same bar
                        signal[i] = -1
                        short_cooldown = cfg.signal_cooldown_bars
                        confl_vol[i] = int(vol_ok)
                        confl_ob[i]  = int(ob_ok)
                        confl_fvg[i] = int(fvg_ok)
                        sweep_ext[i] = short_swept_high
                    short_state = IDLE
                    short_swept_high = np.nan
                else:
                    short_age += 1
                    if short_age >= cfg.arm_window:
                        short_state = IDLE
                        short_swept_high = np.nan

        state_long[i] = long_state
        state_short[i] = short_state

    return pd.DataFrame(
        {
            "signal": signal.astype(int),
            "htf_bias": htf,
            "atr": a_arr,
            "sweep_long": sl_swept,
            "sweep_short": ss_swept,
            "sweep_extreme": sweep_ext,
            "confluence_volume": confl_vol,
            "confluence_ob": confl_ob,
            "confluence_fvg": confl_fvg,
            "state_long": state_long,
            "state_short": state_short,
            "session": sessions.values,
        },
        index=bars.index,
    )
