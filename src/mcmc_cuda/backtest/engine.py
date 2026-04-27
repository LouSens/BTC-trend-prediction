"""Vectorized backtester for a single-instrument signal series.

Inputs:
- close prices (pd.Series, UTC-indexed)
- signal series (-1/0/+1) aligned to close

Conventions:
- Signal at bar t is acted on at bar t+1 (no look-ahead). With bar data we
  use the t+1 close as the fill price approximation; a fill-on-open variant
  is a one-line change once OHLC is wired in.
- Position is held until the signal changes. Transaction cost (spread +
  slippage) is charged on each side of the trade. Position size 1 (signed)
  in price units (USD per oz); account-currency PnL = price PnL * lot_size.
- Swap (overnight rollover) accrues per bar while in position, with the
  daily rate spread evenly across `bars_per_day`.

Output: DataFrame with columns
    close, signal, position, gross_pnl, entry_cost, swap_cost,
    net_pnl, equity, drawdown
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from mcmc_cuda.backtest.costs import CostModel


@dataclass
class BacktestConfig:
    initial_equity: float = 10_000.0      # USD; informational for plotting
    contract_size: float = 100.0          # 1 lot of XAUUSD = 100 oz
    cost: CostModel = field(default_factory=CostModel)


def run_backtest(
    close: pd.Series, signal: pd.Series, cfg: BacktestConfig | None = None
) -> pd.DataFrame:
    cfg = cfg or BacktestConfig()
    df = pd.concat([close.rename("close"), signal.rename("signal")], axis=1).dropna()
    df["signal"] = df["signal"].astype(int)

    # Act on signal one bar later — no look-ahead.
    df["position"] = df["signal"].shift(1).fillna(0).astype(int)

    # Bar PnL in price units per 1 oz, signed by current position.
    price_diff = df["close"].diff().fillna(0.0)
    df["gross_pnl"] = df["position"] * price_diff

    # Entry/exit costs: |Δposition| counts both sides of a flip.
    # 0->1 or 1->0  -> |Δ|=1 -> half a round trip
    # 1->-1         -> |Δ|=2 -> full round trip
    pos_change = df["position"].diff().abs().fillna(0.0)
    df["entry_cost"] = pos_change * cfg.cost.round_trip_cost_price() * 0.5

    # Swap: signed price-unit per oz, per bar. Positive = broker pays us.
    swap_per_bar = df["position"].map(
        {1: cfg.cost.per_bar_swap_price(1), -1: cfg.cost.per_bar_swap_price(-1), 0: 0.0}
    )
    df["swap_cost"] = -swap_per_bar  # convert to "cost" sign convention

    df["net_pnl"] = df["gross_pnl"] - df["entry_cost"] - df["swap_cost"]

    # Scale to USD assuming 1 lot held when in position (sizing in Phase 4).
    df["equity"] = cfg.initial_equity + df["net_pnl"].cumsum() * cfg.contract_size

    rolling_max = df["equity"].cummax()
    df["drawdown"] = (df["equity"] - rolling_max) / rolling_max

    return df


def trade_log(bt: pd.DataFrame) -> pd.DataFrame:
    """Reduce the bar-level frame to one row per round-trip trade."""
    import numpy as np

    pos = bt["position"].values
    changes = list(np.flatnonzero(np.diff(pos, prepend=0) != 0))
    rows = []
    for k, entry_idx in enumerate(changes):
        side = int(pos[entry_idx])
        if side == 0:
            continue
        exit_idx = changes[k + 1] if k + 1 < len(changes) else len(pos) - 1
        seg = bt.iloc[entry_idx:exit_idx + 1]
        rows.append(
            dict(
                entry_time=bt.index[entry_idx],
                exit_time=bt.index[exit_idx],
                side=side,
                entry_price=float(bt["close"].iloc[entry_idx]),
                exit_price=float(bt["close"].iloc[exit_idx]),
                bars=int(exit_idx - entry_idx),
                gross_pnl=float(seg["gross_pnl"].sum()),
                costs=float((seg["entry_cost"] + seg["swap_cost"]).sum()),
                net_pnl=float(seg["net_pnl"].sum()),
            )
        )
    return pd.DataFrame(rows)
