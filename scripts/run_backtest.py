"""End-to-end backtest: bars -> ensemble signal -> filters -> backtest -> metrics + chart.

    # baseline (no filters, close-only engine)
    python scripts/run_backtest.py --years 2 --timeframe M15

    # Phase 2: all filters + ATR-based TP/SL exits + interactive HTML chart
    python scripts/run_backtest.py --years 2 \\
        --use-regime --use-strength --use-slope --use-momentum --tp-sl

Outputs (timestamped) under artifacts/:
- equity_<ts>.png            static equity + drawdown
- trades_<ts>.csv            per-trade log
- metrics_<ts>.json          Sharpe / Sortino / MDD / etc.
- chart_<ts>.html            interactive Plotly chart (open in browser)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import matplotlib
import numpy as np
import typer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mcmc_cuda.backtest.engine import run_backtest, trade_log
from mcmc_cuda.backtest.engine_ohlc import (
    OHLCBacktestConfig,
    run_backtest_ohlc,
    trade_log_ohlc,
)
from mcmc_cuda.backtest.metrics import compute as compute_metrics
from mcmc_cuda.config import ARTIFACTS_DIR
from mcmc_cuda.data.loader import load_bars
from mcmc_cuda.strategy.ensemble import EnsembleConfig, generate_signals
from mcmc_cuda.strategy.filters import FilterConfig, apply_filters, compute_filter_frame
from mcmc_cuda.ui.trade_chart import render_trade_chart

app = typer.Typer(add_completion=False)


@app.command()
def main(
    symbol: str = "XAUUSD",
    timeframe: str = "M15",
    years: float = 2.0,
    horizon: int = 16,
    train_window: int = 2000,
    n_states: int = 5,
    prob_threshold: float = 0.55,
    refit_every: int = 96,
    n_mc_paths: int = 50_000,
    n_markov_paths: int = 50_000,
    # ---- Phase 2 filter flags ----
    use_regime: bool = typer.Option(False, "--use-regime/--no-regime"),
    use_strength: bool = typer.Option(False, "--use-strength/--no-strength"),
    use_slope: bool = typer.Option(False, "--use-slope/--no-slope"),
    use_momentum: bool = typer.Option(False, "--use-momentum/--no-momentum"),
    adx_min: float = 25.0,
    slope_window: int = 20,
    rsi_length: int = 14,
    # ---- Engine + viz ----
    tp_sl: bool = typer.Option(False, "--tp-sl/--no-tp-sl",
                                help="Use OHLC engine with ATR-based TP/SL exits"),
    atr_mult_tp: float = 3.0,
    atr_mult_sl: float = 1.5,
    chart: bool = typer.Option(True, "--chart/--no-chart",
                               help="Write interactive Plotly HTML chart"),
    chart_max_bars: int = 5000,
    invert_signal: bool = typer.Option(
        False, "--invert-signal/--no-invert-signal",
        help="Flip signal sign — useful when the diagnostic shows anti-edge",
    ),
    # ---- Data ----
    csv: str | None = typer.Option(None, help="Local CSV/parquet to use instead of MT5"),
    use_mt5: bool = True,
):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(365 * years))

    print(f"[1/5] Loading {symbol} {timeframe} bars ({start.date()} -> {end.date()})...")
    bars = load_bars(symbol, timeframe, start, end, csv_path=csv, use_mt5=use_mt5)
    print(f"      {len(bars):,} bars loaded.")

    print(f"[2/5] Generating ensemble signals (horizon={horizon}, train_window={train_window})...")
    sig_cfg = EnsembleConfig(
        horizon=horizon, train_window=train_window, n_states=n_states,
        prob_threshold=prob_threshold, refit_every=refit_every,
        n_mc_paths=n_mc_paths, n_markov_paths=n_markov_paths,
    )
    sigs = generate_signals(bars["close"], sig_cfg)
    raw_signal = sigs["signal"]
    if invert_signal:
        raw_signal = -raw_signal
        print("      Signal inverted (--invert-signal).")
    print(f"      Raw signal changes: {int((raw_signal.diff().abs() > 0).sum())}")

    # Diagnostic: signal direction vs realized future return.
    # Positive correlation -> signal is on the right side of the market.
    # Negative correlation -> signal has anti-edge; consider --invert-signal.
    fwd_ret = bars["close"].pct_change(horizon).shift(-horizon)
    sig_active = raw_signal[raw_signal != 0]
    if len(sig_active) > 50:
        corr = float(sig_active.corr(fwd_ret.reindex(sig_active.index)))
        hit = float((np.sign(sig_active) == np.sign(fwd_ret.reindex(sig_active.index))).mean())
        print(f"      Raw-signal vs {horizon}-bar fwd return: corr={corr:+.4f}, "
              f"hit-rate={hit:.3f}  ({'ANTI-EDGE — consider inverting' if corr < -0.01 else 'edge OK' if corr > 0.01 else 'no edge'})")

    print("[3/5] Applying filters...")
    f_cfg = FilterConfig(
        use_regime=use_regime, use_strength=use_strength,
        use_slope=use_slope, use_momentum=use_momentum,
        adx_min=adx_min, slope_window=slope_window, rsi_length=rsi_length,
    )
    active = [n for n, on in [
        ("regime", use_regime), ("strength", use_strength),
        ("slope", use_slope), ("momentum", use_momentum)
    ] if on]
    print(f"      Active filters: {active or '(none)'}")
    if active:
        ff = compute_filter_frame(bars["high"], bars["low"], bars["close"], f_cfg)
        signal = apply_filters(raw_signal, ff, f_cfg)
    else:
        signal = raw_signal
    print(f"      Filtered signal changes: {int((signal.diff().abs() > 0).sum())}")

    print("[4/5] Running backtest...")
    if tp_sl:
        bt_cfg = OHLCBacktestConfig(atr_mult_tp=atr_mult_tp, atr_mult_sl=atr_mult_sl)
        bt = run_backtest_ohlc(bars[["open", "high", "low", "close"]], signal, bt_cfg)
        trades = trade_log_ohlc(bt)
    else:
        bt = run_backtest(bars["close"], signal)
        trades = trade_log(bt)
    metrics = compute_metrics(bt, trades)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    eq_path = ARTIFACTS_DIR / f"equity_{ts}.png"
    trades_path = ARTIFACTS_DIR / f"trades_{ts}.csv"
    metrics_path = ARTIFACTS_DIR / f"metrics_{ts}.json"
    chart_path = ARTIFACTS_DIR / f"chart_{ts}.html"

    print("[5/5] Saving artifacts...")
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax[0].plot(bt.index, bt["equity"])
    ax[0].set_ylabel("Equity (USD)")
    title_bits = [f"{symbol} {timeframe}", f"h={horizon}", f"thresh={prob_threshold}"]
    if active:
        title_bits.append("filters=" + "+".join(active))
    if tp_sl:
        title_bits.append(f"TP/SL={atr_mult_tp}/{atr_mult_sl} ATR")
    ax[0].set_title(" | ".join(title_bits))
    ax[0].grid(alpha=0.3)
    ax[1].fill_between(bt.index, bt["drawdown"] * 100, 0, color="tab:red", alpha=0.4)
    ax[1].set_ylabel("Drawdown (%)")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(eq_path, dpi=110)
    plt.close(fig)

    trades.to_csv(trades_path, index=False)
    metrics_path.write_text(json.dumps(metrics.to_dict(), indent=2, default=str))

    if chart and tp_sl:
        # Trade chart needs OHLC bars + a trades frame with entry_price/exit_price.
        render_trade_chart(
            bars=bars, bt=bt, trades=trades, output_path=chart_path,
            title=" | ".join(title_bits), max_bars=chart_max_bars,
        )

    print()
    print(json.dumps(metrics.to_dict(), indent=2, default=str))
    print(f"\nEquity curve  -> {eq_path}")
    print(f"Trade log     -> {trades_path}")
    print(f"Metrics JSON  -> {metrics_path}")
    if chart and tp_sl:
        print(f"Trade chart   -> {chart_path}    (open in browser)")
    elif chart and not tp_sl:
        print("Note: --chart requires --tp-sl (OHLC engine for entry/exit prices). "
              "Re-run with --tp-sl to get the interactive HTML.")


if __name__ == "__main__":
    app()
