"""Engine break-even / trailing-stop behaviour."""
from __future__ import annotations

import numpy as np
import pandas as pd

from mcmc_cuda.backtest.engine_ohlc import (
    OHLCBacktestConfig,
    run_backtest_ohlc,
    trade_log_ohlc,
)


def _engineered_long_trade(n: int = 60, peak_atr_mult: float = 1.5) -> pd.DataFrame:
    """Build bars where a long entered at bar 1 climbs by peak_atr_mult*ATR
    and then retraces back to entry, so a break-even-armed run scratches
    while a non-BE run hits the original SL."""
    idx = pd.date_range("2024-06-03 08:00", periods=n, freq="15min", tz="UTC")
    # Stable baseline ATR ~ 1.0.
    rng = np.random.default_rng(0)
    base = 2000.0 + rng.normal(0, 0.5, size=n).cumsum() * 0.0  # constant baseline
    high = base + 0.5
    low  = base - 0.5
    close = base.copy()
    open_ = np.concatenate([[base[0]], base[:-1]])

    # Make bars 0..15 stable so ATR settles to ~1.0.
    # Then bar 16 entry, climb to +2 over bars 17..22, fall back to entry by bar 30.
    entry_bar = 16
    open_[entry_bar] = 2000.0
    close[entry_bar] = 2000.5
    high[entry_bar]  = 2000.6
    low[entry_bar]   = 1999.5
    # Climb up to +2 (well beyond breakeven_at_atr=0.7).
    for k, level in enumerate([2001.0, 2001.5, 2002.0, 2002.0, 2001.5, 2001.0], start=17):
        open_[k] = close[k - 1]
        close[k] = level
        high[k]  = level + 0.2
        low[k]   = open_[k] - 0.2
    # Drift back down to entry, then below.
    for k, level in enumerate([2000.5, 2000.0, 1999.6, 1999.4, 1999.2, 1999.0], start=23):
        open_[k] = close[k - 1]
        close[k] = level
        high[k]  = open_[k] + 0.2
        low[k]   = level - 0.2

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close}, index=idx
    )


def test_breakeven_converts_loser_to_scratch():
    bars = _engineered_long_trade()
    sig = pd.Series(0, index=bars.index)
    sig.iloc[15] = 1   # entry on bar 16

    # Without BE: SL=1.0 ATR -> hit when price falls to ~1999.
    cfg_no_be = OHLCBacktestConfig(
        risk_per_trade=0.005, atr_mult_tp=4.0, atr_mult_sl=1.0,
        breakeven_at_atr=0.0, time_stop_bars=40,
    )
    bt_no = run_backtest_ohlc(bars, sig, cfg_no_be)
    trades_no = trade_log_ohlc(bt_no)

    # With BE: once MFE >= 0.7 ATR (~0.7 USD) the SL moves to entry,
    # so when price retraces to 2000 the BE stop scratches the trade.
    cfg_be = OHLCBacktestConfig(
        risk_per_trade=0.005, atr_mult_tp=4.0, atr_mult_sl=1.0,
        breakeven_at_atr=0.7, breakeven_buffer_atr=0.0, time_stop_bars=40,
    )
    bt_be = run_backtest_ohlc(bars, sig, cfg_be)
    trades_be = trade_log_ohlc(bt_be)

    assert not trades_no.empty
    assert not trades_be.empty
    # The BE-armed trade's loss should be substantially smaller (ideally near zero).
    assert trades_be["net_pnl"].iat[0] > trades_no["net_pnl"].iat[0]


def test_trail_stop_locks_in_some_profit():
    bars = _engineered_long_trade()
    sig = pd.Series(0, index=bars.index)
    sig.iloc[15] = 1

    cfg = OHLCBacktestConfig(
        risk_per_trade=0.005, atr_mult_tp=4.0, atr_mult_sl=1.0,
        trail_arm_atr=1.4, trail_distance_atr=0.5,
        breakeven_at_atr=0.0, time_stop_bars=40,
    )
    bt = run_backtest_ohlc(bars, sig, cfg)
    trades = trade_log_ohlc(bt)
    # Trail engaged after MFE >= 1.4 ATR; should exit at trailed SL above entry.
    assert not trades.empty
    # Exit price should be above entry on a long trade (trail locked in profit).
    row = trades.iloc[0]
    assert row["exit_price"] > row["entry_price"]
