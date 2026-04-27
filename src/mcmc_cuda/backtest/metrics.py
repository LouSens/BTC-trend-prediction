"""Performance metrics for a completed backtest.

Annualization assumes the bar-return series, not the trade-level series, so
metrics like Sharpe stay comparable across timeframes / strategies.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class Metrics:
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    n_trades: int
    win_rate: float
    profit_factor: float
    expectancy: float
    avg_bars_in_trade: float
    bankrupt: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _annualization_factor(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 1.0
    median_spacing = pd.Series(index).diff().median()
    if pd.isna(median_spacing) or median_spacing.total_seconds() <= 0:
        return 1.0
    bars_per_year = pd.Timedelta(days=365).total_seconds() / median_spacing.total_seconds()
    return float(bars_per_year)


def compute(bt: pd.DataFrame, trades: pd.DataFrame) -> Metrics:
    eq = bt["equity"]
    rets = eq.pct_change().fillna(0.0)

    ann = _annualization_factor(bt.index)
    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    years = max((bt.index[-1] - bt.index[0]).total_seconds() / (365 * 86400), 1e-9)
    # CAGR is undefined when equity goes <= 0 (account "blew up"). Report NaN
    # rather than a misleading complex number.
    if eq.iloc[-1] > 0 and eq.iloc[0] > 0:
        cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0)
    else:
        cagr = float("nan")

    std = rets.std(ddof=0)
    sharpe = float(np.sqrt(ann) * rets.mean() / std) if std > 0 else 0.0

    downside = rets[rets < 0]
    dstd = downside.std(ddof=0) if not downside.empty else 0.0
    sortino = float(np.sqrt(ann) * rets.mean() / dstd) if dstd > 0 else 0.0

    mdd = float(bt["drawdown"].min())
    if mdd < 0 and not np.isnan(cagr):
        calmar = float(cagr / abs(mdd))
    else:
        calmar = float("nan")
    bankrupt = bool((eq <= 0).any())

    n = len(trades)
    if n == 0:
        return Metrics(
            total_return, cagr, sharpe, sortino, mdd, calmar,
            0, 0.0, 0.0, 0.0, 0.0, bankrupt,
        )

    wins = trades[trades["net_pnl"] > 0]
    losses = trades[trades["net_pnl"] < 0]
    win_rate = float(len(wins) / n)
    gross_win = float(wins["net_pnl"].sum())
    gross_loss = float(-losses["net_pnl"].sum())
    profit_factor = float(gross_win / gross_loss) if gross_loss > 0 else float("inf")
    expectancy = float(trades["net_pnl"].mean())
    avg_bars = float(trades["bars"].mean())

    return Metrics(
        total_return=total_return,
        cagr=cagr,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=mdd,
        calmar=calmar,
        n_trades=n,
        win_rate=win_rate,
        profit_factor=profit_factor,
        expectancy=expectancy,
        avg_bars_in_trade=avg_bars,
        bankrupt=bankrupt,
    )
