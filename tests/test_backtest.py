"""Backtest engine and metrics unit tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from mcmc_cuda.backtest.engine import BacktestConfig, run_backtest, trade_log
from mcmc_cuda.backtest.engine_ohlc import (
    OHLCBacktestConfig,
    run_backtest_ohlc,
    trade_log_ohlc,
)
from mcmc_cuda.backtest.metrics import compute as compute_metrics


def _synth_close(n: int = 500) -> pd.Series:
    rng = np.random.default_rng(0)
    rets = rng.normal(0, 0.001, size=n)
    price = 2000.0 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    return pd.Series(price, index=idx, name="close")


def test_engine_zero_signal_zero_pnl():
    close = _synth_close()
    sig = pd.Series(0, index=close.index)
    bt = run_backtest(close, sig)
    assert (bt["gross_pnl"] == 0).all()
    assert (bt["entry_cost"] == 0).all()
    assert bt["equity"].iloc[-1] == BacktestConfig().initial_equity


def test_engine_constant_long_pnl_matches_price_change():
    close = _synth_close(200)
    sig = pd.Series(1, index=close.index)
    bt = run_backtest(close, sig)
    # Position is 0 on bar 0 (shift) and 1 thereafter; price_diff[0] is also 0.
    # Sum telescopes to close[-1] - close[0].
    expected_gross = close.iloc[-1] - close.iloc[0]
    assert abs(bt["gross_pnl"].sum() - expected_gross) < 1e-6


def test_position_change_charges_cost():
    close = _synth_close(100)
    sig = pd.Series(0, index=close.index)
    sig.iloc[10:60] = 1
    bt = run_backtest(close, sig)
    # Two transitions (0->1 at bar 11, 1->0 at bar 61) -> one round-trip worth.
    n_changes = (bt["position"].diff().abs() > 0).sum()
    assert n_changes == 2
    assert bt["entry_cost"].sum() > 0


def _synth_ohlc(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rets = rng.normal(0, 0.0015, size=n)
    close = 2000.0 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.0005, size=n)))
    low = close * (1 - np.abs(rng.normal(0, 0.0005, size=n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def test_ohlc_engine_zero_signal_no_trades():
    bars = _synth_ohlc(300)
    sig = pd.Series(0, index=bars.index)
    bt = run_backtest_ohlc(bars, sig)
    assert (bt["position"] == 0).all()
    assert bt["equity"].iloc[-1] == OHLCBacktestConfig().initial_equity


def test_ohlc_risk_sizing_caps_loss_at_risk_fraction():
    """A losing trade with risk_per_trade=0.01 should lose ~1% of equity at SL."""
    bars = _synth_ohlc(400)
    sig = pd.Series(0, index=bars.index)
    sig.iloc[50] = 1  # one long trigger
    cfg = OHLCBacktestConfig(
        initial_equity=10_000.0,
        risk_per_trade=0.01,
        atr_mult_tp=1.0,
        atr_mult_sl=0.5,
    )
    bt = run_backtest_ohlc(bars, sig, cfg)
    trades = trade_log_ohlc(bt)
    losers = trades[trades["exit_reason"] == "sl"]
    if not losers.empty:
        # Loss per SL hit shouldn't exceed ~risk fraction (allow slack for costs/spread).
        worst_loss_pct = -losers["net_pnl"].min() / cfg.initial_equity
        assert worst_loss_pct < 0.02, f"SL loss {worst_loss_pct:.4f} exceeds 2x risk budget"


def test_ohlc_initial_equity_passes_through():
    bars = _synth_ohlc(200)
    sig = pd.Series(0, index=bars.index)
    cfg = OHLCBacktestConfig(initial_equity=1_000_000_000.0)
    bt = run_backtest_ohlc(bars, sig, cfg)
    assert bt["equity"].iloc[0] == 1_000_000_000.0
    assert bt["equity"].iloc[-1] == 1_000_000_000.0


def test_metrics_smoke():
    close = _synth_close(300)
    sig = pd.Series(0, index=close.index)
    sig.iloc[20:80] = 1
    sig.iloc[120:170] = -1
    bt = run_backtest(close, sig)
    trades = trade_log(bt)
    m = compute_metrics(bt, trades)
    assert m.n_trades == len(trades) > 0
    assert m.max_drawdown <= 0
    assert isinstance(m.sharpe, float)
