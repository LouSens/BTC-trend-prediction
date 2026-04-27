"""Pull historical XAUUSD bars from MT5 and persist to the parquet cache.

Usage:
    python scripts/pull_history.py --years 5 --timeframe M15
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import typer

from mcmc_cuda.data.mt5_loader import fetch_bars

app = typer.Typer(add_completion=False)


@app.command()
def main(
    symbol: str = "XAUUSD",
    timeframe: str = "M15",
    years: float = 5.0,
    end: str | None = typer.Option(None, help="ISO date for end (default: now UTC)"),
):
    end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc) if end else datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=int(365 * years))
    df = fetch_bars(symbol, timeframe, start_dt, end_dt)
    print(f"Fetched {len(df):,} bars for {symbol} {timeframe}")
    print(f"Range: {df.index.min()} -> {df.index.max()}")


if __name__ == "__main__":
    app()
