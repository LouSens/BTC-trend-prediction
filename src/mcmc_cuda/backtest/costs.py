"""XAUUSD-aware cost model for the vectorized backtester.

XAUUSD specifics that matter for honest backtesting:
- Quoted with 2 decimal digits (point = 0.01); a "pip" colloquially is 0.10.
- Typical retail demo spread: 15-50 points (0.15 - 0.50 USD per oz).
- Contract size: 100 oz per standard lot.
- Swap (overnight rollover) is non-trivial for gold and asymmetric long vs
  short. We treat it per-bar so M15-bar holding still incurs proportional cost.
- Slippage modeled as a fixed-points adverse fill on entry+exit.

Costs are quoted in *price units* (USD per oz), not in account currency, so
the engine can stay currency-agnostic until lot-sizing time.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostModel:
    spread_points: float = 25.0       # half-spread applied on entry and exit
    slippage_points: float = 5.0      # adverse fill in points, per side
    point: float = 0.01               # XAUUSD point size in price units
    swap_long_per_day: float = -7.0   # USD per lot per day, broker-dependent
    swap_short_per_day: float = 2.5
    bars_per_day: int = 96            # M15

    def round_trip_cost_price(self) -> float:
        """Total entry+exit transaction cost in price units (per oz)."""
        return (self.spread_points + 2 * self.slippage_points) * self.point

    def per_bar_swap_price(self, side: int) -> float:
        """Per-bar swap charge in price units. side=+1 long, -1 short.

        We approximate the daily swap as evenly distributed across bars in
        the day. Real brokers credit at rollover, but this stays continuous
        and avoids spurious step changes in equity for the backtest.
        """
        if side == 0:
            return 0.0
        daily = self.swap_long_per_day if side == 1 else self.swap_short_per_day
        return daily / self.bars_per_day / 100.0  # per-oz: lot is 100 oz
