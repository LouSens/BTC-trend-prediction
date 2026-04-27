"""OHLC-aware backtester with ATR-based TP/SL exits and per-bar trade markers.

Why a second engine vs. extending the close-only one:
- Once you introduce intrabar exits the per-bar PnL is no longer just
  position * Δclose — entries/exits happen at TP/SL prices that may differ
  from the bar's close. A separate, tested engine is clearer than overloading.

Exit logic (per bar while in position):
  - long:  if low <= sl_price  -> exit at sl_price
           elif high >= tp_price -> exit at tp_price
           else: continue holding
  - short: symmetric
A pessimistic ordering rule is used: SL is checked before TP within the same
bar (so when a bar's range covers both, we credit the loss). This is the
honest assumption for a backtest without tick data.

Entries: signal at bar t -> entry at bar t+1's open. TP/SL prices are set
relative to entry using ATR (atr_mult_tp, atr_mult_sl).

Output: same column set as engine.run_backtest plus
    entry_price, exit_price, exit_reason  (NaN unless on the bar of the event)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mcmc_cuda.backtest.costs import CostModel
from mcmc_cuda.features.strength import atr


@dataclass
class OHLCBacktestConfig:
    initial_equity: float = 10_000.0
    contract_size: float = 100.0
    atr_length: int = 14
    atr_mult_tp: float = 3.0       # take profit at entry +/- 3*ATR
    atr_mult_sl: float = 1.5       # stop loss at entry +/- 1.5*ATR (R:R = 2)
    cost: CostModel = field(default_factory=CostModel)


def run_backtest_ohlc(
    bars: pd.DataFrame, signal: pd.Series, cfg: OHLCBacktestConfig | None = None
) -> pd.DataFrame:
    """Bars must have columns: open, high, low, close (UTC index)."""
    cfg = cfg or OHLCBacktestConfig()
    required = {"open", "high", "low", "close"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"bars missing OHLC columns: {missing}")

    df = bars.join(signal.rename("signal"), how="inner").dropna(subset=["close", "signal"])
    df["signal"] = df["signal"].astype(int)
    df["atr"] = atr(df["high"], df["low"], df["close"], length=cfg.atr_length)

    n = len(df)
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    sig = df["signal"].values
    a = df["atr"].values

    position = np.zeros(n, dtype=np.int8)
    entry_price = np.full(n, np.nan)
    exit_price = np.full(n, np.nan)
    exit_reason = np.array([""] * n, dtype=object)
    bar_pnl = np.zeros(n)
    cost_per_bar = np.zeros(n)

    cur_pos = 0
    cur_entry = np.nan
    cur_tp = np.nan
    cur_sl = np.nan
    rt_cost = cfg.cost.round_trip_cost_price()

    for i in range(1, n):
        prev_close = c[i - 1]

        # 1. If currently in a position, check intrabar SL then TP.
        if cur_pos != 0:
            tp_hit = (cur_pos == 1 and h[i] >= cur_tp) or (cur_pos == -1 and l[i] <= cur_tp)
            sl_hit = (cur_pos == 1 and l[i] <= cur_sl) or (cur_pos == -1 and h[i] >= cur_sl)

            if sl_hit:
                fill = cur_sl
                bar_pnl[i] = cur_pos * (fill - prev_close)
                cost_per_bar[i] += rt_cost * 0.5  # exit half
                # swap cost for the bar before exit
                cost_per_bar[i] += -cfg.cost.per_bar_swap_price(cur_pos)
                exit_price[i] = fill
                exit_reason[i] = "sl"
                position[i] = cur_pos
                cur_pos, cur_entry, cur_tp, cur_sl = 0, np.nan, np.nan, np.nan
            elif tp_hit:
                fill = cur_tp
                bar_pnl[i] = cur_pos * (fill - prev_close)
                cost_per_bar[i] += rt_cost * 0.5
                cost_per_bar[i] += -cfg.cost.per_bar_swap_price(cur_pos)
                exit_price[i] = fill
                exit_reason[i] = "tp"
                position[i] = cur_pos
                cur_pos, cur_entry, cur_tp, cur_sl = 0, np.nan, np.nan, np.nan
            else:
                # held the full bar
                bar_pnl[i] = cur_pos * (c[i] - prev_close)
                cost_per_bar[i] += -cfg.cost.per_bar_swap_price(cur_pos)
                position[i] = cur_pos

        # 2. If flat after step 1 and prior bar's signal asks for entry, open at this bar's open.
        if cur_pos == 0 and sig[i - 1] != 0 and not np.isnan(a[i - 1]):
            cur_pos = int(sig[i - 1])
            cur_entry = o[i]
            tp_dist = cfg.atr_mult_tp * a[i - 1]
            sl_dist = cfg.atr_mult_sl * a[i - 1]
            cur_tp = cur_entry + cur_pos * tp_dist
            cur_sl = cur_entry - cur_pos * sl_dist
            cost_per_bar[i] += rt_cost * 0.5  # entry half
            entry_price[i] = cur_entry
            position[i] = cur_pos
            # First-bar PnL: from entry (open) to bar close, but only if we
            # didn't already exit above. Re-check TP/SL from open->close.
            # If the same bar that opened also hit TP/SL, exit now.
            tp_hit = (cur_pos == 1 and h[i] >= cur_tp) or (cur_pos == -1 and l[i] <= cur_tp)
            sl_hit = (cur_pos == 1 and l[i] <= cur_sl) or (cur_pos == -1 and h[i] >= cur_sl)
            if sl_hit:
                bar_pnl[i] += cur_pos * (cur_sl - cur_entry)
                cost_per_bar[i] += rt_cost * 0.5
                exit_price[i] = cur_sl
                exit_reason[i] = "sl"
                cur_pos, cur_entry, cur_tp, cur_sl = 0, np.nan, np.nan, np.nan
            elif tp_hit:
                bar_pnl[i] += cur_pos * (cur_tp - cur_entry)
                cost_per_bar[i] += rt_cost * 0.5
                exit_price[i] = cur_tp
                exit_reason[i] = "tp"
                cur_pos, cur_entry, cur_tp, cur_sl = 0, np.nan, np.nan, np.nan
            else:
                bar_pnl[i] += cur_pos * (c[i] - cur_entry)

    df_out = df.copy()
    df_out["position"] = position
    df_out["entry_price"] = entry_price
    df_out["exit_price"] = exit_price
    df_out["exit_reason"] = exit_reason
    df_out["gross_pnl"] = bar_pnl
    df_out["costs"] = cost_per_bar
    df_out["net_pnl"] = bar_pnl - cost_per_bar
    df_out["equity"] = cfg.initial_equity + df_out["net_pnl"].cumsum() * cfg.contract_size

    rolling_max = df_out["equity"].cummax()
    df_out["drawdown"] = (df_out["equity"] - rolling_max) / rolling_max
    return df_out


def trade_log_ohlc(bt: pd.DataFrame) -> pd.DataFrame:
    """Reduce OHLC engine output into one row per round-trip trade."""
    rows = []
    in_trade = False
    entry_idx = None
    side = 0
    for i in range(len(bt)):
        ep = bt["entry_price"].iat[i]
        xp = bt["exit_price"].iat[i]
        pos = int(bt["position"].iat[i])
        if not in_trade and not np.isnan(ep):
            in_trade = True
            entry_idx = i
            side = pos
        if in_trade and not np.isnan(xp):
            seg = bt.iloc[entry_idx:i + 1]
            rows.append(
                dict(
                    entry_time=bt.index[entry_idx],
                    exit_time=bt.index[i],
                    side=side,
                    entry_price=float(bt["entry_price"].iat[entry_idx]),
                    exit_price=float(xp),
                    exit_reason=bt["exit_reason"].iat[i],
                    bars=int(i - entry_idx),
                    gross_pnl=float(seg["gross_pnl"].sum()),
                    costs=float(seg["costs"].sum()),
                    net_pnl=float(seg["net_pnl"].sum()),
                )
            )
            in_trade = False
            entry_idx = None
            side = 0
    return pd.DataFrame(rows)
