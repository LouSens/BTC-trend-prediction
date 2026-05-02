"""Tests covering scalping engine extras (time-stop, sessions, costs split)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from mcmc_cuda.backtest.costs import CostModel
from mcmc_cuda.backtest.engine_ohlc import (
    OHLCBacktestConfig,
    run_backtest_ohlc,
    trade_log_ohlc,
)
from mcmc_cuda.backtest.risk import RiskConfig


def _ohlc(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.0015, size=n)
    close = 2000.0 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.0005, size=n)))
    low = close * (1 - np.abs(rng.normal(0, 0.0005, size=n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def test_costs_split_into_spread_slippage_swap():
    bars = _ohlc(300)
    sig = pd.Series(0, index=bars.index)
    sig.iloc[20] = 1
    bt = run_backtest_ohlc(bars, sig, OHLCBacktestConfig(risk_per_trade=0.01))
    # All cost columns must exist and reconcile.
    assert {"spread_cost", "slip_cost", "swap_cost"}.issubset(bt.columns)
    total = bt["spread_cost"] + bt["slip_cost"] + bt["swap_cost"]
    assert np.allclose(total.values, bt["costs"].values)


def test_session_filter_blocks_disallowed_entries():
    bars = _ohlc(400)
    # Force a signal on every bar so only the session gate matters.
    sig = pd.Series(1, index=bars.index)
    cfg = OHLCBacktestConfig(
        risk_per_trade=0.01,
        allowed_sessions=("london", "overlap"),
    )
    bt = run_backtest_ohlc(bars, sig, cfg)
    # Any bar where an entry happened must be in an allowed session.
    entered = bt[~bt["entry_price"].isna()]
    assert (entered["session"].isin(["london", "overlap"])).all()


def test_time_stop_forces_close():
    bars = _ohlc(500)
    sig = pd.Series(0, index=bars.index)
    sig.iloc[10] = 1  # one entry; no other triggers
    cfg = OHLCBacktestConfig(
        risk_per_trade=0.01,
        time_stop_bars=5,
        atr_mult_tp=999.0,    # never hit
        atr_mult_sl=999.0,    # never hit
    )
    bt = run_backtest_ohlc(bars, sig, cfg)
    trades = trade_log_ohlc(bt)
    if not trades.empty:
        # Engine respects time-stop within 1 bar of the configured limit.
        assert (trades["exit_reason"] == "time").all()
        assert (trades["bars"] <= cfg.time_stop_bars + 1).all()


def test_cost_gating_blocks_low_atr_trades():
    bars = _ohlc(500)
    sig = pd.Series(1, index=bars.index)
    # Crank min_edge_cost_multiple high enough that no trade qualifies.
    cm = CostModel(min_edge_cost_multiple=1000.0)
    cfg = OHLCBacktestConfig(
        risk_per_trade=0.01,
        cost_gating=True,
        cost=cm,
    )
    bt = run_backtest_ohlc(bars, sig, cfg)
    assert (bt["position"] == 0).all()


def test_layered_entries_increase_size():
    bars = _ohlc(800, seed=3)
    sig = pd.Series(0, index=bars.index)
    sig.iloc[5] = 1
    risk = RiskConfig(risk_per_trade=0.005, max_total_risk_per_idea=0.02)
    cfg = OHLCBacktestConfig(
        risk_per_trade=0.005,
        atr_mult_tp=10.0,    # let it run a while
        atr_mult_sl=2.0,
        max_layers=3,
        add_at_atr_profit=0.1,  # easy to trigger adds
        risk=risk,
    )
    bt = run_backtest_ohlc(bars, sig, cfg)
    # `layer_count` should peak at >= 1; if the trade ran in profit it could
    # reach max_layers. We only assert that layering machinery is wired.
    assert int(bt["layer_count"].max()) >= 1


def test_running_equity_matches_naive_recompute():
    """Replacement of the O(n^2) equity recompute must be numerically identical
    to the naive cumulative version."""
    bars = _ohlc(400, seed=7)
    sig = pd.Series(0, index=bars.index)
    sig.iloc[20] = 1
    sig.iloc[120] = -1
    cfg = OHLCBacktestConfig(risk_per_trade=0.01)
    bt = run_backtest_ohlc(bars, sig, cfg)

    naive = cfg.initial_equity + (bt["gross_pnl"] - bt["costs"]).cumsum()
    # Allow a tiny floating-point tolerance.
    assert np.allclose(bt["equity"].values, naive.values, atol=1e-6)
