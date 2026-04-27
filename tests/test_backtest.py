"""Backtest engine and metrics unit tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from mcmc_cuda.backtest.engine import BacktestConfig, run_backtest, trade_log
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
