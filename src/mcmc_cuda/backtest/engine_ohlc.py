"""OHLC-aware backtester with ATR-based TP/SL exits and risk-based sizing.

Sizing modes:
- Fixed lot:   `risk_per_trade <= 0`  -> hold `contract_size` oz whenever in
               a position. Same as the original behavior.
- Risk-based:  `risk_per_trade > 0`   -> at entry, size the position so that
               hitting the stop loses exactly `risk_per_trade` of current
               equity. This is the standard "risk 1% per trade" sizing used
               in retail trading and is what survives drawdowns.

PnL is denominated directly in USD (account currency). Equity = initial
equity + cumulative net PnL.

Exit logic (per bar while in position):
  - long:  if low <= sl_price  -> exit at sl_price
           elif high >= tp_price -> exit at tp_price
  - short: symmetric
SL is checked before TP within the same bar (pessimistic without tick data).

Entries: signal at bar t -> entry at bar t+1's open. TP/SL prices are set
relative to entry using ATR (atr_mult_tp, atr_mult_sl).

Output columns:
    open, high, low, close, signal, atr,
    position (signed -1/0/1), size_oz (signed oz held this bar),
    entry_price, exit_price, exit_reason,
    gross_pnl, costs, net_pnl, equity, drawdown
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pandas as pd

from mcmc_cuda.backtest.costs import CostModel
from mcmc_cuda.features.strength import atr


@dataclass
class OHLCBacktestConfig:
    initial_equity: float = 10_000.0
    contract_size: float = 100.0       # oz when risk_per_trade <= 0
    risk_per_trade: float = 0.0        # fraction of equity to risk per trade
    max_lot_oz: float = 1e9            # safety cap on position size
    atr_length: int = 14
    atr_mult_tp: float = 3.0
    atr_mult_sl: float = 1.5
    cost: CostModel = field(default_factory=CostModel)


def _prepare(bars: pd.DataFrame, signal: pd.Series, cfg: OHLCBacktestConfig) -> pd.DataFrame:
    required = {"open", "high", "low", "close"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"bars missing OHLC columns: {missing}")
    df = bars.join(signal.rename("signal"), how="inner").dropna(subset=["close", "signal"])
    df["signal"] = df["signal"].astype(int)
    df["atr"] = atr(df["high"], df["low"], df["close"], length=cfg.atr_length)
    return df


def _step_state(cfg: OHLCBacktestConfig):
    """Per-bar mutable state shared between run and iter variants."""
    return dict(
        cur_pos=0,
        cur_size=0.0,
        cur_entry=np.nan,
        cur_tp=np.nan,
        cur_sl=np.nan,
        equity=float(cfg.initial_equity),
        peak_equity=float(cfg.initial_equity),
    )


def _size_for_entry(side: int, entry: float, sl: float, equity: float, cfg: OHLCBacktestConfig) -> float:
    """Return signed oz size for a new entry."""
    if cfg.risk_per_trade > 0:
        sl_dist = abs(entry - sl)
        if sl_dist <= 0 or not np.isfinite(sl_dist):
            return 0.0
        size = (cfg.risk_per_trade * max(equity, 0.0)) / sl_dist
        size = min(size, cfg.max_lot_oz)
        return float(side) * size
    return float(side) * cfg.contract_size


def run_backtest_ohlc(
    bars: pd.DataFrame, signal: pd.Series, cfg: OHLCBacktestConfig | None = None
) -> pd.DataFrame:
    cfg = cfg or OHLCBacktestConfig()
    df = _prepare(bars, signal, cfg)
    n = len(df)
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    sig = df["signal"].values
    a = df["atr"].values

    position = np.zeros(n, dtype=np.int8)
    size_oz = np.zeros(n)
    entry_price = np.full(n, np.nan)
    exit_price = np.full(n, np.nan)
    exit_reason = np.array([""] * n, dtype=object)
    bar_pnl = np.zeros(n)         # USD
    cost_per_bar = np.zeros(n)    # USD
    equity_arr = np.full(n, cfg.initial_equity, dtype=float)
    drawdown = np.zeros(n)

    rt_cost_per_oz = cfg.cost.round_trip_cost_price()  # USD/oz round-trip
    swap_per_bar_long = -cfg.cost.per_bar_swap_price(1)   # cost sign convention
    swap_per_bar_short = -cfg.cost.per_bar_swap_price(-1)

    st = _step_state(cfg)

    for i in range(n):
        prev_close = c[i - 1] if i > 0 else c[0]

        # 1) If in a position, check intrabar exits.
        if st["cur_pos"] != 0:
            cur_pos = st["cur_pos"]
            cur_size = st["cur_size"]
            cur_tp = st["cur_tp"]
            cur_sl = st["cur_sl"]
            tp_hit = (cur_pos == 1 and h[i] >= cur_tp) or (cur_pos == -1 and l[i] <= cur_tp)
            sl_hit = (cur_pos == 1 and l[i] <= cur_sl) or (cur_pos == -1 and h[i] >= cur_sl)
            swap = swap_per_bar_long if cur_pos == 1 else swap_per_bar_short

            if sl_hit:
                fill = cur_sl
                bar_pnl[i] = cur_size * (fill - prev_close)
                cost_per_bar[i] += rt_cost_per_oz * 0.5 * abs(cur_size) + swap * abs(cur_size)
                exit_price[i] = fill
                exit_reason[i] = "sl"
                position[i] = cur_pos
                size_oz[i] = cur_size
                st["cur_pos"] = 0
                st["cur_size"] = 0.0
                st["cur_entry"] = st["cur_tp"] = st["cur_sl"] = np.nan
            elif tp_hit:
                fill = cur_tp
                bar_pnl[i] = cur_size * (fill - prev_close)
                cost_per_bar[i] += rt_cost_per_oz * 0.5 * abs(cur_size) + swap * abs(cur_size)
                exit_price[i] = fill
                exit_reason[i] = "tp"
                position[i] = cur_pos
                size_oz[i] = cur_size
                st["cur_pos"] = 0
                st["cur_size"] = 0.0
                st["cur_entry"] = st["cur_tp"] = st["cur_sl"] = np.nan
            else:
                bar_pnl[i] = cur_size * (c[i] - prev_close)
                cost_per_bar[i] += swap * abs(cur_size)
                position[i] = cur_pos
                size_oz[i] = cur_size

        # 2) If flat after step 1 and prior bar's signal asks for entry, open at this bar's open.
        if st["cur_pos"] == 0 and i > 0 and sig[i - 1] != 0 and not np.isnan(a[i - 1]):
            side = int(sig[i - 1])
            entry = o[i]
            tp_dist = cfg.atr_mult_tp * a[i - 1]
            sl_dist = cfg.atr_mult_sl * a[i - 1]
            tp = entry + side * tp_dist
            sl = entry - side * sl_dist
            # equity at this point: initial + cumulative net so far (excl. this bar's entry cost)
            equity_now = cfg.initial_equity + (bar_pnl[: i + 1].sum() - cost_per_bar[: i + 1].sum())
            sized = _size_for_entry(side, entry, sl, equity_now, cfg)
            if sized != 0.0:
                st["cur_pos"] = side
                st["cur_size"] = sized
                st["cur_entry"] = entry
                st["cur_tp"] = tp
                st["cur_sl"] = sl
                cost_per_bar[i] += rt_cost_per_oz * 0.5 * abs(sized)
                entry_price[i] = entry
                position[i] = side
                size_oz[i] = sized
                # Same-bar TP/SL check from open->close
                tp_hit = (side == 1 and h[i] >= tp) or (side == -1 and l[i] <= tp)
                sl_hit = (side == 1 and l[i] <= sl) or (side == -1 and h[i] >= sl)
                if sl_hit:
                    bar_pnl[i] += sized * (sl - entry)
                    cost_per_bar[i] += rt_cost_per_oz * 0.5 * abs(sized)
                    exit_price[i] = sl
                    exit_reason[i] = "sl"
                    st["cur_pos"] = 0
                    st["cur_size"] = 0.0
                    st["cur_entry"] = st["cur_tp"] = st["cur_sl"] = np.nan
                elif tp_hit:
                    bar_pnl[i] += sized * (tp - entry)
                    cost_per_bar[i] += rt_cost_per_oz * 0.5 * abs(sized)
                    exit_price[i] = tp
                    exit_reason[i] = "tp"
                    st["cur_pos"] = 0
                    st["cur_size"] = 0.0
                    st["cur_entry"] = st["cur_tp"] = st["cur_sl"] = np.nan
                else:
                    bar_pnl[i] += sized * (c[i] - entry)

        net_i = bar_pnl[i] - cost_per_bar[i]
        st["equity"] += net_i
        if st["equity"] > st["peak_equity"]:
            st["peak_equity"] = st["equity"]
        equity_arr[i] = st["equity"]
        drawdown[i] = (st["equity"] - st["peak_equity"]) / st["peak_equity"] if st["peak_equity"] > 0 else 0.0

    df_out = df.copy()
    df_out["position"] = position
    df_out["size_oz"] = size_oz
    df_out["entry_price"] = entry_price
    df_out["exit_price"] = exit_price
    df_out["exit_reason"] = exit_reason
    df_out["gross_pnl"] = bar_pnl
    df_out["costs"] = cost_per_bar
    df_out["net_pnl"] = bar_pnl - cost_per_bar
    df_out["equity"] = equity_arr
    df_out["drawdown"] = drawdown
    return df_out


def iter_backtest_ohlc(
    bars: pd.DataFrame, signal: pd.Series, cfg: OHLCBacktestConfig | None = None
) -> Iterator[dict]:
    """Yield one event dict per bar — used by the live chart for streaming playback.

    Each event has: time, open, high, low, close, position, size_oz,
    entry_price (NaN if no entry on this bar), exit_price (NaN if no exit),
    exit_reason, equity, drawdown, tp, sl.
    """
    full = run_backtest_ohlc(bars, signal, cfg)
    for ts, row in full.iterrows():
        yield {
            "time": ts,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "position": int(row["position"]),
            "size_oz": float(row["size_oz"]),
            "entry_price": float(row["entry_price"]) if not np.isnan(row["entry_price"]) else np.nan,
            "exit_price": float(row["exit_price"]) if not np.isnan(row["exit_price"]) else np.nan,
            "exit_reason": str(row["exit_reason"]),
            "equity": float(row["equity"]),
            "drawdown": float(row["drawdown"]),
        }


def trade_log_ohlc(bt: pd.DataFrame) -> pd.DataFrame:
    """Reduce OHLC engine output into one row per round-trip trade."""
    rows = []
    in_trade = False
    entry_idx = None
    side = 0
    size = 0.0
    for i in range(len(bt)):
        ep = bt["entry_price"].iat[i]
        xp = bt["exit_price"].iat[i]
        pos = int(bt["position"].iat[i])
        if not in_trade and not np.isnan(ep):
            in_trade = True
            entry_idx = i
            side = pos
            size = float(bt["size_oz"].iat[i]) if "size_oz" in bt.columns else float(pos)
        if in_trade and not np.isnan(xp):
            seg = bt.iloc[entry_idx:i + 1]
            rows.append(
                dict(
                    entry_time=bt.index[entry_idx],
                    exit_time=bt.index[i],
                    side=side,
                    size_oz=abs(size),
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
            size = 0.0
    return pd.DataFrame(rows)
