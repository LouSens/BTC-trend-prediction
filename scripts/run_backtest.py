"""End-to-end backtest: bars -> ensemble signal -> filters -> backtest -> metrics + chart.

Examples:

    # baseline (no filters, close-only engine)
    python scripts/run_backtest.py --years 2 --timeframe M15

    # Phase 2: filters + ATR TP/SL exits + risk-based sizing (1% per trade)
    python scripts/run_backtest.py --years 2 \\
        --use-regime --use-strength --use-slope --use-momentum \\
        --tp-sl --risk-per-trade 0.01 --initial-equity 1000000000

    # Scalping preset (M5, tight TP/SL, fast refit)
    python scripts/run_backtest.py --scalp --tp-sl --risk-per-trade 0.01 --live

Outputs (timestamped) under artifacts/:
- equity_<ts>.png            static equity + drawdown
- trades_<ts>.csv            per-trade log
- metrics_<ts>.json          Sharpe / Sortino / MDD / etc.
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

app = typer.Typer(add_completion=False)


SCALP_DEFAULTS = dict(
    timeframe="M5",
    horizon=4,
    train_window=1500,
    refit_every=48,
    atr_mult_tp=1.0,
    atr_mult_sl=0.5,
    prob_threshold=0.55,
)


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
    # ---- Engine ----
    tp_sl: bool = typer.Option(False, "--tp-sl/--no-tp-sl",
                                help="Use OHLC engine with ATR-based TP/SL exits"),
    atr_mult_tp: float = 3.0,
    atr_mult_sl: float = 1.5,
    initial_equity: float = typer.Option(10_000.0, "--initial-equity",
                                         help="Starting account balance in USD."),
    risk_per_trade: float = typer.Option(0.0, "--risk-per-trade",
                                         help="Fraction of equity to risk per trade (0 = fixed lot). e.g. 0.01 = 1%%."),
    contract_size: float = typer.Option(100.0, "--contract-size",
                                        help="Oz held when risk-per-trade=0 (1 standard lot = 100 oz)."),
    max_lot_oz: float = typer.Option(1e9, "--max-lot-oz",
                                     help="Hard cap on position size in oz."),
    # ---- Scalping preset ----
    scalp: bool = typer.Option(
        False, "--scalp/--no-scalp",
        help="Apply scalping defaults: M5, horizon=4, ATR TP/SL=1.0/0.5, fast refit. "
             "Override-able by passing the individual flags after --scalp.",
    ),
    # ---- Live playback ----
    live: bool = typer.Option(False, "--live/--no-live",
                              help="Open a desktop window and animate the backtest as it plays out."),
    live_speed: int = typer.Option(2, "--live-speed",
                                   help="Bars advanced per animation frame. Higher = faster playback."),
    live_interval_ms: int = typer.Option(20, "--live-interval-ms",
                                         help="Delay between frames in ms."),
    invert_signal: bool = typer.Option(
        False, "--invert-signal/--no-invert-signal",
        help="Flip signal sign — useful when the diagnostic shows anti-edge",
    ),
    # ---- Data ----
    csv: str | None = typer.Option(None, help="Local CSV/parquet to use instead of MT5"),
    use_mt5: bool = True,
):
    if scalp:
        # Only override values the user didn't explicitly set (typer gives us
        # the defaults so we can't easily detect "untouched"; we apply scalp
        # defaults unconditionally — the user can still pass flags AFTER
        # --scalp on the same command line to override.).
        timeframe = SCALP_DEFAULTS["timeframe"] if timeframe == "M15" else timeframe
        horizon = SCALP_DEFAULTS["horizon"] if horizon == 16 else horizon
        train_window = SCALP_DEFAULTS["train_window"] if train_window == 2000 else train_window
        refit_every = SCALP_DEFAULTS["refit_every"] if refit_every == 96 else refit_every
        atr_mult_tp = SCALP_DEFAULTS["atr_mult_tp"] if atr_mult_tp == 3.0 else atr_mult_tp
        atr_mult_sl = SCALP_DEFAULTS["atr_mult_sl"] if atr_mult_sl == 1.5 else atr_mult_sl
        print(f"[scalp] timeframe={timeframe} horizon={horizon} "
              f"refit_every={refit_every} TP/SL={atr_mult_tp}/{atr_mult_sl} ATR")

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
        bt_cfg = OHLCBacktestConfig(
            initial_equity=initial_equity,
            contract_size=contract_size,
            risk_per_trade=risk_per_trade,
            max_lot_oz=max_lot_oz,
            atr_mult_tp=atr_mult_tp,
            atr_mult_sl=atr_mult_sl,
        )
        if live:
            from mcmc_cuda.ui.live_chart import LiveChartConfig, play_live
            print("      [live] opening playback window — close it to continue.")
            live_cfg = LiveChartConfig(
                interval_ms=live_interval_ms,
                bars_per_frame=live_speed,
                title=f"{symbol} {timeframe} | risk={risk_per_trade*100:.2f}% | "
                      f"TP/SL={atr_mult_tp}/{atr_mult_sl} ATR | "
                      f"start ${initial_equity:,.0f}",
            )
            bt = play_live(bars[["open", "high", "low", "close"]], signal, bt_cfg, live_cfg)
        else:
            bt = run_backtest_ohlc(bars[["open", "high", "low", "close"]], signal, bt_cfg)
        trades = trade_log_ohlc(bt)
    else:
        if live:
            print("      [live] requires --tp-sl (the close-only engine has no entry/exit prices). "
                  "Falling back to non-live.")
        bt = run_backtest(bars["close"], signal)
        trades = trade_log(bt)
    metrics = compute_metrics(bt, trades)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    eq_path = ARTIFACTS_DIR / f"equity_{ts}.png"
    trades_path = ARTIFACTS_DIR / f"trades_{ts}.csv"
    metrics_path = ARTIFACTS_DIR / f"metrics_{ts}.json"

    print("[5/5] Saving artifacts...")
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax[0].plot(bt.index, bt["equity"])
    ax[0].set_ylabel("Equity (USD)")
    title_bits = [f"{symbol} {timeframe}", f"h={horizon}", f"thresh={prob_threshold}"]
    if active:
        title_bits.append("filters=" + "+".join(active))
    if tp_sl:
        title_bits.append(f"TP/SL={atr_mult_tp}/{atr_mult_sl} ATR")
        if risk_per_trade > 0:
            title_bits.append(f"risk={risk_per_trade*100:.2f}%/trade")
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

    print()
    print(json.dumps(metrics.to_dict(), indent=2, default=str))
    print(f"\nEquity curve  -> {eq_path}")
    print(f"Trade log     -> {trades_path}")
    print(f"Metrics JSON  -> {metrics_path}")


if __name__ == "__main__":
    app()
