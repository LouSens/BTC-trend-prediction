"""Interactive HTML trade chart: price candlesticks + entry/exit markers + equity curve.

Outputs a self-contained HTML file you can double-click to open in a browser.
You can scroll, zoom, hover for trade details (entry time, side, entry price,
exit price, TP/SL hit, net PnL).

Layout (vertical stack):
  Row 1: candlestick of price with TP/SL horizontal lines per trade and
         green-up / red-down markers at entries; ▲/▼ markers at exits coloured
         by win (green) / loss (red). Hover shows full trade card.
  Row 2: equity curve.
  Row 3: drawdown shaded.

If you have matplotlib already in the run, the static PNG from the existing
script is still produced — this is the *interactive* counterpart.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render_trade_chart(
    bars: pd.DataFrame,
    bt: pd.DataFrame,
    trades: pd.DataFrame,
    output_path: str | Path,
    title: str = "XAUUSD MCMC Backtest",
    max_bars: int | None = 5000,
) -> Path:
    """Write an interactive HTML chart to `output_path`.

    `bars`: OHLC frame (index = UTC datetime).
    `bt`:   backtest output frame (must contain 'equity' and 'drawdown').
    `trades`: trade log (entry_time, exit_time, side, entry_price, exit_price,
              exit_reason, net_pnl).
    `max_bars`: cap displayed bars for chart responsiveness; None = all.
    """
    output_path = Path(output_path)

    if max_bars is not None and len(bars) > max_bars:
        bars = bars.iloc[-max_bars:]
        bt = bt.loc[bt.index >= bars.index[0]]
        trades = trades[trades["entry_time"] >= bars.index[0]] if not trades.empty else trades

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.25, 0.15], vertical_spacing=0.03,
        subplot_titles=("Price + Trades", "Equity", "Drawdown"),
    )

    # --- Candlesticks ---
    fig.add_trace(
        go.Candlestick(
            x=bars.index, open=bars["open"], high=bars["high"],
            low=bars["low"], close=bars["close"],
            name="XAUUSD", showlegend=False,
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        ),
        row=1, col=1,
    )

    # --- Trade markers ---
    if trades is not None and not trades.empty:
        # Cast to plain Python types; some Plotly versions choke on int8/object
        # customdata mixes when one row is empty.
        trades = trades.copy()
        trades["side"] = trades["side"].astype(int)
        trades["entry_price"] = trades["entry_price"].astype(float)
        trades["exit_price"] = trades["exit_price"].astype(float)
        trades["net_pnl"] = trades["net_pnl"].astype(float)
        trades["exit_reason"] = trades["exit_reason"].astype(str).fillna("")
        trades = trades.dropna(subset=["entry_time", "exit_time", "entry_price", "exit_price"])

        long_e = trades[trades["side"] == 1]
        short_e = trades[trades["side"] == -1]
        wins = trades[trades["net_pnl"] > 0]
        losses = trades[trades["net_pnl"] <= 0]

        fig.add_trace(go.Scatter(
            x=long_e["entry_time"], y=long_e["entry_price"],
            mode="markers", name="Long entry",
            marker=dict(symbol="triangle-up", size=11, color="#00c853",
                        line=dict(width=1, color="white")),
            hovertemplate=(
                "<b>LONG entry</b><br>%{x}<br>price=%{y:.2f}<extra></extra>"
            ),
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=short_e["entry_time"], y=short_e["entry_price"],
            mode="markers", name="Short entry",
            marker=dict(symbol="triangle-down", size=11, color="#d50000",
                        line=dict(width=1, color="white")),
            hovertemplate=(
                "<b>SHORT entry</b><br>%{x}<br>price=%{y:.2f}<extra></extra>"
            ),
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=wins["exit_time"], y=wins["exit_price"],
            mode="markers", name="Exit (win)",
            marker=dict(symbol="circle", size=8, color="#00c853",
                        line=dict(width=1, color="black")),
            customdata=wins[["side", "entry_price", "exit_reason", "net_pnl"]].values,
            hovertemplate=(
                "<b>WIN exit (%{customdata[2]})</b><br>%{x}<br>"
                "entry=%{customdata[1]:.2f} → exit=%{y:.2f}<br>"
                "side=%{customdata[0]}<br>net_pnl=%{customdata[3]:.4f}<extra></extra>"
            ),
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=losses["exit_time"], y=losses["exit_price"],
            mode="markers", name="Exit (loss)",
            marker=dict(symbol="x", size=10, color="#d50000",
                        line=dict(width=2)),
            customdata=losses[["side", "entry_price", "exit_reason", "net_pnl"]].values,
            hovertemplate=(
                "<b>LOSS exit (%{customdata[2]})</b><br>%{x}<br>"
                "entry=%{customdata[1]:.2f} → exit=%{y:.2f}<br>"
                "side=%{customdata[0]}<br>net_pnl=%{customdata[3]:.4f}<extra></extra>"
            ),
        ), row=1, col=1)

        # Light line connecting each trade's entry to its exit.
        seg_x: list = []
        seg_y: list = []
        for _, t in trades.iterrows():
            seg_x.extend([t["entry_time"], t["exit_time"], None])
            seg_y.extend([t["entry_price"], t["exit_price"], None])
        fig.add_trace(go.Scatter(
            x=seg_x, y=seg_y, mode="lines", name="Trade path",
            line=dict(color="rgba(120,120,120,0.45)", width=1, dash="dot"),
            hoverinfo="skip", showlegend=False,
        ), row=1, col=1)

    # --- Equity ---
    fig.add_trace(go.Scatter(
        x=bt.index, y=bt["equity"], mode="lines", name="Equity",
        line=dict(color="#1976d2", width=1.5),
        hovertemplate="%{x}<br>equity=$%{y:,.2f}<extra></extra>",
    ), row=2, col=1)

    # --- Drawdown ---
    fig.add_trace(go.Scatter(
        x=bt.index, y=bt["drawdown"] * 100,
        mode="lines", name="Drawdown",
        line=dict(color="#c62828", width=1),
        fill="tozeroy", fillcolor="rgba(198,40,40,0.18)",
        hovertemplate="%{x}<br>dd=%{y:.2f}%<extra></extra>",
    ), row=3, col=1)

    fig.update_layout(
        title=title,
        height=950,
        showlegend=True,
        legend=dict(orientation="h", y=1.05),
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Price (USD/oz)", row=1, col=1)
    fig.update_yaxes(title_text="USD", row=2, col=1)
    fig.update_yaxes(title_text="%", row=3, col=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Inline plotly.js so the chart works without internet / behind proxies.
    # Costs ~3.5 MB per HTML file but eliminates the "black screen" failure
    # mode where the CDN fetch is blocked.
    fig.write_html(str(output_path), include_plotlyjs=True, full_html=True)
    return output_path
