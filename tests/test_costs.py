"""CostModel session-aware behaviour and gating helpers."""
from __future__ import annotations

import math

from mcmc_cuda.backtest.costs import CostModel


def test_session_spread_overrides_static():
    cm = CostModel(spread_points=25.0)
    static = cm.round_trip_cost_price()
    overlap = cm.round_trip_cost_price("overlap")
    asia = cm.round_trip_cost_price("asia")
    # Asia must be more expensive than overlap.
    assert asia > overlap
    # Static is the no-session fallback.
    assert math.isclose(static, (25.0 + 2 * 5.0) * 0.01, rel_tol=1e-9)


def test_disable_session_uses_static_spread():
    cm = CostModel(use_session_spread=False)
    a = cm.round_trip_cost_price("asia")
    b = cm.round_trip_cost_price("overlap")
    assert math.isclose(a, b, rel_tol=1e-12)


def test_total_expected_cost_includes_swap():
    cm = CostModel(swap_long_per_day=-9.6, swap_short_per_day=2.4, bars_per_day=96)
    rt = cm.round_trip_cost_price("london")
    # Long, 96 bars (one full day) -> rt + |daily_swap_long| / 100 (per oz)
    total_long = cm.total_expected_trade_cost_price(side=1, holding_bars=96, session="london")
    assert total_long > rt
    # Short: positive swap is a credit, should not increase cost
    total_short = cm.total_expected_trade_cost_price(side=-1, holding_bars=96, session="london")
    assert math.isclose(total_short, rt, abs_tol=1e-9)


def test_min_edge_required_scales_with_multiplier():
    cm = CostModel(min_edge_cost_multiple=2.0)
    rt = cm.round_trip_cost_price("overlap")
    req = cm.min_edge_required_price(side=1, holding_bars=0, session="overlap")
    assert math.isclose(req, 2.0 * rt, rel_tol=1e-9)


def test_entry_exit_cost_split_balances_round_trip():
    cm = CostModel()
    e = cm.entry_cost_price(side=1, session="london")
    x = cm.exit_cost_price(side=1, session="london")
    rt = cm.round_trip_cost_price("london")
    assert math.isclose(e + x, rt, rel_tol=1e-12)


def test_asymmetric_slippage():
    cm = CostModel(slippage_points=5.0, slippage_long_extra=2.0, slippage_short_extra=0.0)
    long_e = cm.entry_cost_price(side=1, session="overlap")
    short_e = cm.entry_cost_price(side=-1, session="overlap")
    assert long_e > short_e
