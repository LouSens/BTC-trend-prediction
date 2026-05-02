"""Walk-forward / rolling evaluation harness.

The harness slices a bar/signal frame into successive windows of
[train | validate | test] and records out-of-sample metrics on each test
slice. The strategy callable controls what "training" means — for a
parametric strategy this might be a refit of the Markov chain; for the
breakout-retest strategy, training is a no-op and only the rolling test
windows matter for honest reporting.

Returns a DataFrame with one row per fold containing the headline metrics
plus the train/validate/test boundaries — easy to plot or aggregate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from mcmc_cuda.backtest.engine_ohlc import (
    OHLCBacktestConfig,
    run_backtest_ohlc,
    trade_log_ohlc,
)
from mcmc_cuda.backtest.metrics import compute as compute_metrics


SignalFn = Callable[[pd.DataFrame, pd.DataFrame], pd.Series]
"""Signal callable: (train_bars, eval_bars) -> signal series indexed like eval_bars."""


@dataclass
class WalkForwardConfig:
    train_bars: int = 4_000
    val_bars: int = 1_000
    test_bars: int = 2_000
    step_bars: int = 2_000     # how far to advance each fold
    min_test_trades: int = 5   # warn when a fold is too sparse to interpret


def walk_forward(
    bars: pd.DataFrame,
    signal_fn: SignalFn,
    bt_cfg: OHLCBacktestConfig | None = None,
    wf_cfg: WalkForwardConfig | None = None,
) -> pd.DataFrame:
    """Run a rolling walk-forward backtest.

    `signal_fn(train, eval)` is invoked once per fold to produce the test
    signal. The train slice is provided so parametric strategies can refit;
    rule-based strategies can ignore it.

    Returns one row per fold with columns:
        train_start, train_end, val_start, val_end, test_start, test_end,
        n_trades, sharpe, sortino, max_drawdown, calmar, profit_factor,
        win_rate, expectancy, costs_pct_gross, total_return.
    """
    bt_cfg = bt_cfg or OHLCBacktestConfig()
    wf_cfg = wf_cfg or WalkForwardConfig()
    n = len(bars)
    win = wf_cfg.train_bars + wf_cfg.val_bars + wf_cfg.test_bars
    if n < win:
        raise ValueError(f"Need at least {win} bars for one fold; got {n}")

    rows: list[dict] = []
    start = 0
    while start + win <= n:
        train_end = start + wf_cfg.train_bars
        val_end   = train_end + wf_cfg.val_bars
        test_end  = val_end + wf_cfg.test_bars
        train = bars.iloc[start:train_end]
        eval_ = bars.iloc[train_end:test_end]
        # Generate signal across train_end..test_end.
        signal = signal_fn(train, eval_)
        signal = signal.reindex(eval_.index).fillna(0).astype(int)

        # Use only the test slice for evaluation (validation is reserved for
        # downstream model selection — we simply skip it here).
        test_bars_slice = eval_.iloc[wf_cfg.val_bars:]
        test_signal = signal.iloc[wf_cfg.val_bars:]

        bt = run_backtest_ohlc(
            test_bars_slice[["open", "high", "low", "close"]],
            test_signal,
            bt_cfg,
        )
        trades = trade_log_ohlc(bt)
        m = compute_metrics(bt, trades)

        gross = float(bt["gross_pnl"].abs().sum())
        costs = float(bt["costs"].sum()) if "costs" in bt.columns else 0.0
        costs_pct = costs / gross if gross > 0 else float("nan")

        rows.append(dict(
            train_start=bars.index[start],
            train_end=bars.index[train_end - 1],
            val_start=bars.index[train_end],
            val_end=bars.index[val_end - 1],
            test_start=bars.index[val_end],
            test_end=bars.index[test_end - 1],
            n_trades=m.n_trades,
            sharpe=m.sharpe,
            sortino=m.sortino,
            max_drawdown=m.max_drawdown,
            calmar=m.calmar,
            profit_factor=m.profit_factor,
            win_rate=m.win_rate,
            expectancy=m.expectancy,
            avg_bars_in_trade=m.avg_bars_in_trade,
            costs_pct_gross=costs_pct,
            total_return=m.total_return,
        ))
        start += wf_cfg.step_bars

    return pd.DataFrame(rows)
