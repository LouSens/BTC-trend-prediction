"""End-to-end backtest: load bars -> ensemble signal -> backtest -> metrics + plot.

    python scripts/run_backtest.py --years 2 --timeframe M15
    python scripts/run_backtest.py --csv path/to/XAUUSD_M15.csv --no-mt5

Outputs:
- prints metrics to stdout
- saves equity curve PNG to artifacts/equity_<timestamp>.png
- saves trade log CSV to artifacts/trades_<timestamp>.csv
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import matplotlib
import typer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mcmc_cuda.backtest.engine import BacktestConfig, run_backtest, trade_log
from mcmc_cuda.backtest.metrics import compute as compute_metrics
from mcmc_cuda.config import ARTIFACTS_DIR
from mcmc_cuda.data.loader import load_bars
from mcmc_cuda.strategy.ensemble import EnsembleConfig, generate_signals

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
    csv: str | None = typer.Option(None, help="Local CSV/parquet to use instead of MT5"),
    use_mt5: bool = True,
):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(365 * years))

    print(f"[1/4] Loading {symbol} {timeframe} bars ({start.date()} -> {end.date()})...")
    bars = load_bars(symbol, timeframe, start, end, csv_path=csv, use_mt5=use_mt5)
    print(f"      {len(bars):,} bars loaded.")

    print(f"[2/4] Generating ensemble signals (horizon={horizon}, train_window={train_window})...")
    sig_cfg = EnsembleConfig(
        horizon=horizon,
        train_window=train_window,
        n_states=n_states,
        prob_threshold=prob_threshold,
        refit_every=refit_every,
    )
    sigs = generate_signals(bars["close"], sig_cfg)
    n_trades_est = (sigs["signal"].diff().abs() > 0).sum()
    print(f"      Signal changes: {int(n_trades_est)}")

    print("[3/4] Running backtest...")
    bt = run_backtest(bars["close"], sigs["signal"])
    trades = trade_log(bt)
    metrics = compute_metrics(bt, trades)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    eq_path = ARTIFACTS_DIR / f"equity_{ts}.png"
    trades_path = ARTIFACTS_DIR / f"trades_{ts}.csv"
    metrics_path = ARTIFACTS_DIR / f"metrics_{ts}.json"

    print("[4/4] Plotting and saving artifacts...")
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax[0].plot(bt.index, bt["equity"])
    ax[0].set_ylabel("Equity (USD)")
    ax[0].set_title(f"{symbol} {timeframe} | h={horizon} | thresh={prob_threshold}")
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
