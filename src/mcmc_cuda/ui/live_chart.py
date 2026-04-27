"""Live matplotlib playback of an OHLC backtest.

Opens a desktop window and animates through the bars one-by-one. You see
candles forming, ▲/▼ entry markers as trades open, and ●/✕ exit markers as
they close, with the equity and drawdown panels updating live.

Implementation notes:
- We pre-compute the backtest with `run_backtest_ohlc`, then play the result
  back frame-by-frame with `matplotlib.animation.FuncAnimation`. This keeps
  the engine pure (still fully tested headless) and avoids stalls in the UI
  thread doing math.
- Backend: the default interactive backend on Windows is TkAgg, which ships
  with the Python installer. No extra packages required.
- We render line price (close) rather than candlesticks for performance —
  per-bar candle artists become slow above a few thousand frames.
"""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation

from mcmc_cuda.backtest.engine_ohlc import OHLCBacktestConfig, run_backtest_ohlc


@dataclass
class LiveChartConfig:
    interval_ms: int = 30           # delay between animated frames
    bars_per_frame: int = 1         # advance N bars per tick (>1 = faster playback)
    window_bars: int = 400          # rolling window of bars shown on the price panel
    title: str = "XAUUSD MCMC Live Backtest"


def play_live(
    bars: pd.DataFrame,
    signal: pd.Series,
    bt_cfg: OHLCBacktestConfig | None = None,
    live_cfg: LiveChartConfig | None = None,
) -> pd.DataFrame:
    """Run the OHLC backtest and animate it. Returns the full backtest frame."""
    bt_cfg = bt_cfg or OHLCBacktestConfig()
    live_cfg = live_cfg or LiveChartConfig()

    # Force an interactive backend for the live window.
    if matplotlib.get_backend().lower() in {"agg", "module://matplotlib_inline.backend_inline"}:
        try:
            matplotlib.use("TkAgg", force=True)
        except Exception:
            pass

    bt = run_backtest_ohlc(bars, signal, bt_cfg)

    times = bt.index.to_numpy()
    close = bt["close"].to_numpy()
    equity = bt["equity"].to_numpy()
    dd_pct = bt["drawdown"].to_numpy() * 100.0
    entry_price = bt["entry_price"].to_numpy()
    exit_price = bt["exit_price"].to_numpy()
    side_at_entry = bt["position"].to_numpy()
    exit_reasons = bt["exit_reason"].to_numpy()
    # Net PnL per bar to colour exits by win/loss using the trade's last bar.
    net_pnl = bt["net_pnl"].to_numpy()

    long_mask = (~np.isnan(entry_price)) & (side_at_entry == 1)
    short_mask = (~np.isnan(entry_price)) & (side_at_entry == -1)
    exit_mask = ~np.isnan(exit_price)
    win_exit_mask = exit_mask & (net_pnl >= 0)
    loss_exit_mask = exit_mask & (net_pnl < 0)

    fig, (ax_p, ax_e, ax_d) = plt.subplots(
        3, 1, figsize=(13, 8), sharex=False,
        gridspec_kw=dict(height_ratios=[3, 1.5, 1]),
    )
    fig.suptitle(live_cfg.title, fontsize=12)

    (price_line,) = ax_p.plot([], [], color="#1976d2", linewidth=1.0, label="close")
    long_scatter = ax_p.scatter([], [], marker="^", s=70, color="#00c853",
                                edgecolor="white", linewidth=0.6, label="long entry", zorder=5)
    short_scatter = ax_p.scatter([], [], marker="v", s=70, color="#d50000",
                                 edgecolor="white", linewidth=0.6, label="short entry", zorder=5)
    win_scatter = ax_p.scatter([], [], marker="o", s=42, color="#00c853",
                               edgecolor="black", linewidth=0.5, label="win exit", zorder=5)
    loss_scatter = ax_p.scatter([], [], marker="x", s=55, color="#d50000",
                                linewidth=2.0, label="loss exit", zorder=5)
    tp_line = ax_p.axhline(np.nan, color="#00c853", linestyle="--", linewidth=0.8, alpha=0.0)
    sl_line = ax_p.axhline(np.nan, color="#d50000", linestyle="--", linewidth=0.8, alpha=0.0)
    ax_p.set_ylabel("Price (USD/oz)")
    ax_p.grid(alpha=0.25)
    ax_p.legend(loc="upper left", fontsize=8, ncol=5)

    (eq_line,) = ax_e.plot([], [], color="#1976d2", linewidth=1.2)
    ax_e.set_ylabel("Equity (USD)")
    ax_e.grid(alpha=0.25)

    (dd_line,) = ax_d.plot([], [], color="#c62828", linewidth=0.9)
    dd_fill = [ax_d.fill_between([], [], 0, color="#c62828", alpha=0.25)]
    ax_d.set_ylabel("Drawdown (%)")
    ax_d.set_xlabel("bar")
    ax_d.grid(alpha=0.25)

    status_txt = fig.text(
        0.99, 0.965, "", ha="right", va="top",
        fontsize=10, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fafafa", edgecolor="#cccccc"),
    )

    n_total = len(bt)
    step = max(1, int(live_cfg.bars_per_frame))
    n_frames = (n_total + step - 1) // step

    state = {"in_pos": False, "tp": np.nan, "sl": np.nan}

    def _update(frame_idx: int):
        i = min((frame_idx + 1) * step, n_total)
        x = np.arange(i)

        lo = max(0, i - live_cfg.window_bars)
        win_x = np.arange(lo, i)
        price_line.set_data(win_x, close[lo:i])
        if i > lo:
            ax_p.set_xlim(lo, max(lo + 5, i + 5))
            ymin = float(np.nanmin(close[lo:i]))
            ymax = float(np.nanmax(close[lo:i]))
            pad = max((ymax - ymin) * 0.05, 0.5)
            ax_p.set_ylim(ymin - pad, ymax + pad)

        def _pts(mask: np.ndarray, prices: np.ndarray) -> np.ndarray:
            idx = np.flatnonzero(mask[:i])
            if idx.size == 0:
                return np.empty((0, 2))
            keep = idx[idx >= lo]
            if keep.size == 0:
                return np.empty((0, 2))
            return np.column_stack([keep, prices[keep]])

        long_scatter.set_offsets(_pts(long_mask, entry_price))
        short_scatter.set_offsets(_pts(short_mask, entry_price))
        win_scatter.set_offsets(_pts(win_exit_mask, exit_price))
        loss_scatter.set_offsets(_pts(loss_exit_mask, exit_price))

        # TP/SL guide lines while in position
        last = i - 1
        if last >= 0:
            if not np.isnan(entry_price[last]):
                state["in_pos"] = True
                # Recover tp/sl from the engine output's adjacent bars: easier
                # to recompute from atr if needed, but bt frame already has it
                # implicit. We just show entry as a dotted guide.
            if not np.isnan(exit_price[last]):
                state["in_pos"] = False
                tp_line.set_alpha(0.0)
                sl_line.set_alpha(0.0)

        eq_line.set_data(x, equity[:i])
        ax_e.set_xlim(0, max(50, i))
        if i > 0:
            ymin = float(np.nanmin(equity[:i]))
            ymax = float(np.nanmax(equity[:i]))
            pad = max((ymax - ymin) * 0.05, 1.0)
            ax_e.set_ylim(ymin - pad, ymax + pad)

        dd_line.set_data(x, dd_pct[:i])
        ax_d.set_xlim(0, max(50, i))
        if i > 0:
            ax_d.set_ylim(min(float(np.nanmin(dd_pct[:i])), -0.1) * 1.1, 0.5)
        # Refresh fill
        if dd_fill[0] is not None:
            dd_fill[0].remove()
        dd_fill[0] = ax_d.fill_between(x, dd_pct[:i], 0, color="#c62828", alpha=0.25)

        cur_eq = equity[i - 1] if i > 0 else bt_cfg.initial_equity
        cur_dd = dd_pct[i - 1] if i > 0 else 0.0
        ts = pd.Timestamp(times[i - 1]).strftime("%Y-%m-%d %H:%M") if i > 0 else ""
        n_trades = int(np.isnan(exit_price[:i]).sum() if False else (~np.isnan(exit_price[:i])).sum())
        status_txt.set_text(
            f"bar {i}/{n_total}  {ts}\n"
            f"equity ${cur_eq:,.2f}\n"
            f"drawdown {cur_dd:+.2f}%\n"
            f"trades closed {n_trades}"
        )
        return (price_line, long_scatter, short_scatter, win_scatter, loss_scatter,
                eq_line, dd_line, status_txt)

    anim = FuncAnimation(
        fig, _update, frames=n_frames,
        interval=live_cfg.interval_ms, blit=False, repeat=False,
    )
    # Hold a reference so the animation isn't garbage-collected.
    fig._live_anim = anim  # type: ignore[attr-defined]
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()
    return bt
