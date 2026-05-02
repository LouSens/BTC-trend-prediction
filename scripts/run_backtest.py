"""End-to-end backtest: bars -> signal -> filters -> backtest -> metrics + chart.

Two strategies live behind one CLI:

    --strategy ensemble  (default)  : Markov + bootstrap MC ensemble
    --strategy scalp                 : M15 bias + M5 breakout-retest

Examples:

    # Ensemble baseline (close-only engine, no filters)
    python scripts/run_backtest.py --years 2 --timeframe M15

    # Ensemble with filters + ATR TP/SL + 1% risk
    python scripts/run_backtest.py --years 2 \\
        --use-regime --use-strength --use-slope --use-momentum \\
        --tp-sl --risk-per-trade 0.01

    # Scalp v2 on M5, session-filtered, cost-gated, time-stop, layered
    python scripts/run_backtest.py --strategy scalp --timeframe M5 --tp-sl \\
        --risk-per-trade 0.01 --session-filter london,overlap \\
        --cost-gating --time-stop-bars 24 --max-layers 2

Outputs (timestamped) under artifacts/:
- equity_<ts>.png        static equity + drawdown
- trades_<ts>.csv        per-trade log (incl. session, R-multiple, MAE/MFE)
- metrics_<ts>.json      headline + per-session + costs%
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

from mcmc_cuda.backtest.costs import CostModel
from mcmc_cuda.backtest.engine import run_backtest, trade_log
from mcmc_cuda.backtest.engine_ohlc import (
    OHLCBacktestConfig,
    run_backtest_ohlc,
    trade_log_ohlc,
)
from mcmc_cuda.backtest.metrics import compute as compute_metrics
from mcmc_cuda.backtest.metrics import compute_extended
from mcmc_cuda.backtest.risk import RiskConfig
from mcmc_cuda.config import ARTIFACTS_DIR
from mcmc_cuda.data.loader import load_bars
from mcmc_cuda.strategy.ensemble import EnsembleConfig, generate_signals
from mcmc_cuda.strategy.filters import FilterConfig, apply_filters, compute_filter_frame
from mcmc_cuda.strategy.scalp import ScalpConfig, generate_scalp_signals, resample_for_htf
from mcmc_cuda.strategy.smc import SMCConfig, generate_smc_signals

app = typer.Typer(add_completion=False)


# Sensible scalping presets. R/R is wider than the textbook 2:1 because XAUUSD
# friction (round-trip spread+slippage) is large relative to ATR on these
# timeframes; a tight SL gets eaten by costs even at 50% win rate.
M5_SCALP_DEFAULTS = dict(
    timeframe="M5",
    horizon=4,
    train_window=1500,
    refit_every=48,
    atr_mult_tp=2.5,             # wider TP — the move must clear friction
    atr_mult_sl=1.0,
    prob_threshold=0.55,
    time_stop_bars=24,           # 2h on M5
)

M15_SCALP_DEFAULTS = dict(
    timeframe="M15",
    horizon=8,
    atr_mult_tp=3.0,             # 3:1 R/R; SL=1.0 ATR is robust to wicks
    atr_mult_sl=1.0,
    time_stop_bars=16,           # 4h on M15
)


# SMC defaults are tuned for win rate, not pure profit. Wider SL beyond the
# sweep extreme + break-even mover at +0.6 ATR is what pushes win rate up,
# at the cost of fewer trades and lower per-trade expectancy. Tune with
# care; the friction-vs-stop ratio in diagnostics tells you when defaults
# stop being sensible for your timeframe.
SMC_DEFAULTS = dict(
    timeframe="M15",
    atr_mult_tp=2.0,
    atr_mult_sl=1.5,             # SL beyond sweep extreme is naturally wider
    time_stop_bars=20,
    breakeven_at_atr=1.2,         # move SL to BE once +1.2 ATR in profit
    trail_arm_atr=1.6,            # trail after +1.6 ATR (well past BE)
    trail_distance_atr=0.8,
)


# "High win-rate" preset — explicitly trades reverse R/R (TP closer than SL)
# to maximize win rate. NOTE: this is selling tail risk. Expectancy is
# preserved only if signal quality is high enough; one losing streak in a
# regime change wipes out many small wins. Provided because the user asked
# for high WR; not recommended as a long-term production setup.
SMC_HIGH_WR_DEFAULTS = dict(
    timeframe="M15",
    atr_mult_tp=0.8,             # tight TP -> often hit
    atr_mult_sl=2.5,              # wide SL -> rarely hit
    time_stop_bars=24,
    breakeven_at_atr=0.0,         # no BE (would scratch winners)
    trail_arm_atr=0.0,            # no trail
)


@app.command()
def main(
    symbol: str = "XAUUSD",
    timeframe: str = "M15",
    years: float = 2.0,
    strategy: str = typer.Option(
        "ensemble", "--strategy",
        help="Signal source: ensemble | scalp",
    ),
    horizon: int = 3,
    train_window: int = 200,
    n_states: int = 5,
    prob_threshold: float = 0.55,
    refit_every: int = 24,
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
    initial_equity: float = typer.Option(10_000.0, "--initial-equity"),
    risk_per_trade: float = typer.Option(0.0, "--risk-per-trade",
                                         help="Fraction of equity per trade. 0 = fixed lot."),
    contract_size: float = typer.Option(100.0, "--contract-size"),
    max_lot_oz: float = typer.Option(1e9, "--max-lot-oz"),
    # ---- Scalping engine additions ----
    time_stop_bars: int = typer.Option(0, "--time-stop-bars",
                                       help="Force-close after N bars (0 = off)."),
    session_filter: str = typer.Option(
        "", "--session-filter",
        help="Comma list of allowed sessions: asia,london,overlap,ny,dead. Empty = no filter.",
    ),
    cost_gating: bool = typer.Option(
        False, "--cost-gating/--no-cost-gating",
        help="Skip trades whose expected move doesn't beat friction by min_edge_cost_multiple.",
    ),
    min_edge_mult: float = typer.Option(1.5, "--min-edge-mult"),
    max_layers: int = typer.Option(1, "--max-layers",
                                    help="Max pyramid legs per idea. 1 = no layering."),
    add_at_atr_profit: float = typer.Option(0.5, "--add-at-atr-profit"),
    same_bar_tiebreak: str = typer.Option(
        "by_close", "--same-bar-tiebreak",
        help="When SL+TP both touched on a bar: by_close|sl_first|tp_first.",
    ),
    breakeven_at_atr: float = typer.Option(
        0.0, "--breakeven-at-atr",
        help="Move SL to entry once MFE >= N*ATR. 0 disables.",
    ),
    breakeven_buffer_atr: float = typer.Option(0.05, "--breakeven-buffer-atr"),
    trail_arm_atr: float = typer.Option(
        0.0, "--trail-arm-atr",
        help="Arm trailing stop after MFE >= N*ATR. 0 disables.",
    ),
    trail_distance_atr: float = typer.Option(1.0, "--trail-distance-atr"),
    # ---- SMC strategy params ----
    smc_sweep_lookback: int = typer.Option(20, "--smc-sweep-lookback"),
    smc_arm_window: int = typer.Option(5, "--smc-arm-window"),
    smc_vol_z: float = typer.Option(1.5, "--smc-vol-z"),
    smc_require_volume: bool = typer.Option(True, "--smc-vol/--no-smc-vol"),
    smc_require_ob_fvg: bool = typer.Option(True, "--smc-ob/--no-smc-ob"),
    smc_ob_impulse: float = typer.Option(1.2, "--smc-ob-impulse"),
    # ---- Risk ----
    max_daily_loss: float = typer.Option(0.03, "--max-daily-loss"),
    max_consecutive_losses: int = typer.Option(4, "--max-consecutive-losses"),
    cooldown_bars: int = typer.Option(24, "--cooldown-bars"),
    max_total_risk_per_idea: float = typer.Option(0.02, "--max-total-risk-per-idea"),
    min_atr_to_cost_ratio: float = typer.Option(3.0, "--min-atr-to-cost-ratio"),
    min_stop_to_cost_ratio: float = typer.Option(2.0, "--min-stop-to-cost-ratio"),
    # ---- Scalp v2 strategy ----
    scalp_swing_lookback: int = typer.Option(20, "--scalp-swing"),
    scalp_breakout_window: int = typer.Option(10, "--scalp-breakout-window"),
    scalp_retest_window: int = typer.Option(8, "--scalp-retest-window"),
    scalp_retest_atr_pct: float = typer.Option(0.5, "--scalp-retest-atr"),
    scalp_htf: str = typer.Option(
        "15min", "--scalp-htf",
        help="Pandas resample alias for HTF bias (e.g. 15min, 30min).",
    ),
    # ---- Presets ----
    scalp: bool = typer.Option(
        False, "--scalp/--no-scalp",
        help="Apply M5 scalping defaults to the ensemble path.",
    ),
    scalp_m15: bool = typer.Option(
        False, "--scalp-m15/--no-scalp-m15",
        help="Apply M15 scalping defaults.",
    ),
    smc_preset: bool = typer.Option(
        False, "--smc-preset/--no-smc-preset",
        help="Apply SMC scalping preset: --strategy smc, M15, ATR TP/SL=2/1.5, "
             "BE at +1.2 ATR, trail at +1.6 ATR. Override individual flags after.",
    ),
    smc_high_wr: bool = typer.Option(
        False, "--smc-high-wr/--no-smc-high-wr",
        help="High-win-rate SMC preset (reverse R/R: tight TP, wide SL, no BE/trail). "
             "Sells tail risk; honest about it.",
    ),
    # ---- Live playback ----
    live: bool = typer.Option(False, "--live/--no-live"),
    live_speed: int = typer.Option(2, "--live-speed"),
    live_interval_ms: int = typer.Option(20, "--live-interval-ms"),
    invert_signal: bool = typer.Option(False, "--invert-signal/--no-invert-signal"),
    # ---- Data ----
    csv: str | None = typer.Option(None, help="Local CSV/parquet to use instead of MT5"),
    use_mt5: bool = True,
):
    if scalp:
        timeframe       = M5_SCALP_DEFAULTS["timeframe"] if timeframe == "M15" else timeframe
        horizon         = M5_SCALP_DEFAULTS["horizon"] if horizon == 16 else horizon
        train_window    = M5_SCALP_DEFAULTS["train_window"] if train_window == 2000 else train_window
        refit_every     = M5_SCALP_DEFAULTS["refit_every"] if refit_every == 96 else refit_every
        atr_mult_tp     = M5_SCALP_DEFAULTS["atr_mult_tp"] if atr_mult_tp == 3.0 else atr_mult_tp
        atr_mult_sl     = M5_SCALP_DEFAULTS["atr_mult_sl"] if atr_mult_sl == 1.5 else atr_mult_sl
        time_stop_bars  = M5_SCALP_DEFAULTS["time_stop_bars"] if time_stop_bars == 0 else time_stop_bars
        print(f"[scalp/M5] tf={timeframe} h={horizon} TP/SL={atr_mult_tp}/{atr_mult_sl} ATR "
              f"time_stop={time_stop_bars}")

    if scalp_m15:
        timeframe       = M15_SCALP_DEFAULTS["timeframe"] if timeframe == "M15" else timeframe
        horizon         = M15_SCALP_DEFAULTS["horizon"] if horizon == 16 else horizon
        atr_mult_tp     = M15_SCALP_DEFAULTS["atr_mult_tp"] if atr_mult_tp == 3.0 else atr_mult_tp
        atr_mult_sl     = M15_SCALP_DEFAULTS["atr_mult_sl"] if atr_mult_sl == 1.5 else atr_mult_sl
        time_stop_bars  = M15_SCALP_DEFAULTS["time_stop_bars"] if time_stop_bars == 0 else time_stop_bars
        print(f"[scalp/M15] tf={timeframe} h={horizon} TP/SL={atr_mult_tp}/{atr_mult_sl} ATR "
              f"time_stop={time_stop_bars}")

    if smc_preset:
        strategy        = "smc"
        timeframe       = SMC_DEFAULTS["timeframe"] if timeframe == "M15" else timeframe
        atr_mult_tp     = SMC_DEFAULTS["atr_mult_tp"] if atr_mult_tp == 3.0 else atr_mult_tp
        atr_mult_sl     = SMC_DEFAULTS["atr_mult_sl"] if atr_mult_sl == 1.5 else atr_mult_sl
        time_stop_bars  = SMC_DEFAULTS["time_stop_bars"] if time_stop_bars == 0 else time_stop_bars
        breakeven_at_atr   = (
            SMC_DEFAULTS["breakeven_at_atr"] if breakeven_at_atr == 0.0 else breakeven_at_atr
        )
        trail_arm_atr      = (
            SMC_DEFAULTS["trail_arm_atr"] if trail_arm_atr == 0.0 else trail_arm_atr
        )
        trail_distance_atr = (
            SMC_DEFAULTS["trail_distance_atr"] if trail_distance_atr == 1.0 else trail_distance_atr
        )
        print(f"[smc-preset] strategy=smc tf={timeframe} TP/SL={atr_mult_tp}/{atr_mult_sl} ATR "
              f"BE@+{breakeven_at_atr} ATR trail@+{trail_arm_atr} ATR")

    if smc_high_wr:
        strategy        = "smc"
        timeframe       = SMC_HIGH_WR_DEFAULTS["timeframe"] if timeframe == "M15" else timeframe
        atr_mult_tp     = SMC_HIGH_WR_DEFAULTS["atr_mult_tp"] if atr_mult_tp == 3.0 else atr_mult_tp
        atr_mult_sl     = SMC_HIGH_WR_DEFAULTS["atr_mult_sl"] if atr_mult_sl == 1.5 else atr_mult_sl
        time_stop_bars  = SMC_HIGH_WR_DEFAULTS["time_stop_bars"] if time_stop_bars == 0 else time_stop_bars
        breakeven_at_atr = SMC_HIGH_WR_DEFAULTS["breakeven_at_atr"]
        trail_arm_atr   = SMC_HIGH_WR_DEFAULTS["trail_arm_atr"]
        print(f"[smc-high-wr] WARNING: reverse R/R preset ({atr_mult_tp}TP/{atr_mult_sl}SL). "
              f"Sells tail risk for win rate.")

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(365 * years))

    print(f"[1/5] Loading {symbol} {timeframe} bars ({start.date()} -> {end.date()})...")
    bars = load_bars(symbol, timeframe, start, end, csv_path=csv, use_mt5=use_mt5)
    print(f"      {len(bars):,} bars loaded.")

    # ----------------------------------------------------------------------
    # 2) Signal generation
    # ----------------------------------------------------------------------
    if strategy == "scalp":
        print(f"[2/5] Generating scalp signals (HTF={scalp_htf})...")
        htf_bars = resample_for_htf(bars, htf=scalp_htf) if scalp_htf else None
        s_cfg = ScalpConfig(
            swing_lookback=scalp_swing_lookback,
            breakout_window=scalp_breakout_window,
            retest_window=scalp_retest_window,
            retest_atr_pct=scalp_retest_atr_pct,
        )
        scalp_df = generate_scalp_signals(
            bars[["open", "high", "low", "close"]],
            htf_close=htf_bars["close"] if htf_bars is not None else None,
            cfg=s_cfg,
        )
        raw_signal = scalp_df["signal"]
    elif strategy == "smc":
        print(f"[2/5] Generating SMC signals (HTF={scalp_htf})...")
        if "tick_volume" not in bars.columns:
            print("      [WARN] no tick_volume in data — disabling volume-spike filter.")
            smc_require_volume = False
        htf_bars = resample_for_htf(bars, htf=scalp_htf) if scalp_htf else None
        smc_cfg = SMCConfig(
            sweep_lookback=smc_sweep_lookback,
            arm_window=smc_arm_window,
            vol_spike_z=smc_vol_z,
            require_volume_spike=smc_require_volume,
            require_ob_or_fvg=smc_require_ob_fvg,
            ob_impulse_atr_mult=smc_ob_impulse,
        )
        smc_in = bars.copy()
        smc_df = generate_smc_signals(
            smc_in,
            htf_close=htf_bars["close"] if htf_bars is not None else None,
            cfg=smc_cfg,
        )
        raw_signal = smc_df["signal"]
        # Confluence-quality summary upfront.
        fired = smc_df[smc_df["signal"] != 0]
        if not fired.empty:
            print(
                f"      Confluence on {len(fired)} firings: "
                f"vol={int(fired.confluence_volume.sum())} "
                f"OB={int(fired.confluence_ob.sum())} "
                f"FVG={int(fired.confluence_fvg.sum())}"
            )
    else:
        print(f"[2/5] Generating ensemble signals (h={horizon}, train={train_window})...")
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
        diag = (
            "ANTI-EDGE — consider --invert-signal" if corr < -0.01
            else "edge OK" if corr > 0.01 else "no edge"
        )
        print(f"      Signal vs {horizon}-bar fwd return: corr={corr:+.4f}, "
              f"hit-rate={hit:.3f}  ({diag})")

    # ----------------------------------------------------------------------
    # 3) Filters (only meaningful for ensemble; scalp has its own filters)
    # ----------------------------------------------------------------------
    if strategy != "scalp":
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
    else:
        print("[3/5] Filters skipped (scalp strategy embeds its own).")
        active = []
        signal = raw_signal
    print(f"      Filtered signal changes: {int((signal.diff().abs() > 0).sum())}")

    # ----------------------------------------------------------------------
    # 4) Backtest
    # ----------------------------------------------------------------------
    print("[4/5] Running backtest...")
    if tp_sl:
        # Bars-per-day for swap accrual (CostModel default is M15 = 96).
        bars_per_day = max(1, int(round(86400 / (
            (bars.index[-1] - bars.index[0]).total_seconds() / max(1, len(bars) - 1)
        ))))

        cost = CostModel(
            bars_per_day=bars_per_day,
            min_edge_cost_multiple=min_edge_mult,
        )
        risk_cfg = RiskConfig(
            risk_per_trade=risk_per_trade,
            max_total_risk_per_idea=max_total_risk_per_idea,
            max_daily_loss=max_daily_loss,
            max_consecutive_losses=max_consecutive_losses,
            cooldown_bars=cooldown_bars,
            min_atr_to_cost_ratio=min_atr_to_cost_ratio,
            min_stop_to_cost_ratio=min_stop_to_cost_ratio,
            max_lot_oz=max_lot_oz,
            contract_size=contract_size,
        )
        allowed = tuple(s.strip() for s in session_filter.split(",") if s.strip()) if session_filter else ()
        bt_cfg = OHLCBacktestConfig(
            initial_equity=initial_equity,
            contract_size=contract_size,
            risk_per_trade=risk_per_trade,
            max_lot_oz=max_lot_oz,
            atr_mult_tp=atr_mult_tp,
            atr_mult_sl=atr_mult_sl,
            time_stop_bars=time_stop_bars,
            allowed_sessions=allowed,
            cost_gating=cost_gating,
            max_layers=max_layers,
            add_at_atr_profit=add_at_atr_profit,
            same_bar_tiebreak=same_bar_tiebreak,
            breakeven_at_atr=breakeven_at_atr,
            breakeven_buffer_atr=breakeven_buffer_atr,
            trail_arm_atr=trail_arm_atr,
            trail_distance_atr=trail_distance_atr,
            cost=cost,
            risk=risk_cfg,
        )

        # Honest diagnostic: how big is friction relative to the SL distance
        # the strategy is using? If friction approaches SL, the strategy is
        # structurally a coin-flip with negative expectancy.
        from mcmc_cuda.features.strength import atr as _atr
        atr_now = float(_atr(bars["high"], bars["low"], bars["close"]).iloc[-200:].median())
        sl_pts = atr_mult_sl * atr_now
        rt_cost_pts = cost.round_trip_cost_price("overlap")
        friction_pct_sl = rt_cost_pts / sl_pts if sl_pts > 0 else float("inf")
        print(f"      [friction] ATR(median, last 200) = {atr_now:.2f}  "
              f"SL = {sl_pts:.2f}  RT cost (overlap) = {rt_cost_pts:.2f}  "
              f"friction/SL = {friction_pct_sl:.0%}")
        if friction_pct_sl > 0.5:
            print(f"      [WARN] friction is {friction_pct_sl:.0%} of SL distance — "
                  f"this timeframe is cost-dominated. Consider widening atr_mult_sl, "
                  f"using a higher timeframe, or a tighter-spread venue.")
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
            print("      [live] requires --tp-sl. Falling back to non-live.")
        bt = run_backtest(bars["close"], signal)
        trades = trade_log(bt)
    metrics = compute_metrics(bt, trades)
    extended = compute_extended(bt, trades)

    # Why did the strategy close? Mix of reasons tells you what's wrong.
    if not trades.empty and "exit_reason" in trades.columns:
        reason_counts = trades["exit_reason"].value_counts().to_dict()
        same_bar_pct = float((trades["bars"] == 0).mean()) if "bars" in trades.columns else 0.0
        print(f"      [exits] reasons={reason_counts}  same-bar exits={same_bar_pct:.1%}")
        if same_bar_pct > 0.4:
            print(f"      [WARN] {same_bar_pct:.0%} of trades exit on the entry bar — "
                  f"SL is too tight relative to bar range; widen atr_mult_sl.")

    # ----------------------------------------------------------------------
    # 5) Artifacts
    # ----------------------------------------------------------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    eq_path = ARTIFACTS_DIR / f"equity_{ts}.png"
    trades_path = ARTIFACTS_DIR / f"trades_{ts}.csv"
    metrics_path = ARTIFACTS_DIR / f"metrics_{ts}.json"

    print("[5/5] Saving artifacts...")
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax[0].plot(bt.index, bt["equity"])
    ax[0].set_ylabel("Equity (USD)")
    title_bits = [f"{symbol} {timeframe}", strategy, f"h={horizon}"]
    if active:
        title_bits.append("filters=" + "+".join(active))
    if tp_sl:
        title_bits.append(f"TP/SL={atr_mult_tp}/{atr_mult_sl} ATR")
        if risk_per_trade > 0:
            title_bits.append(f"risk={risk_per_trade*100:.2f}%/trade")
        if time_stop_bars > 0:
            title_bits.append(f"ts={time_stop_bars}b")
        if session_filter:
            title_bits.append(f"sess={session_filter}")
    ax[0].set_title(" | ".join(title_bits))
    ax[0].grid(alpha=0.3)
    ax[1].fill_between(bt.index, bt["drawdown"] * 100, 0, color="tab:red", alpha=0.4)
    ax[1].set_ylabel("Drawdown (%)")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(eq_path, dpi=110)
    plt.close(fig)

    trades.to_csv(trades_path, index=False)
    metrics_path.write_text(json.dumps(extended.to_dict(), indent=2, default=str))

    print()
    print(json.dumps(extended.to_dict(), indent=2, default=str))
    print(f"\nEquity curve  -> {eq_path}")
    print(f"Trade log     -> {trades_path}")
    print(f"Metrics JSON  -> {metrics_path}")


if __name__ == "__main__":
    app()
