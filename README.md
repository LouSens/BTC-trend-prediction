# mcmc_cuda

GPU-accelerated Monte Carlo + Markov Chain price prediction for **XAUUSD**, with regime / strength / slope / momentum filters, a vectorized backtester, and an MT5 paper-trading bridge.

> **Paper-trading only.** The MT5 bridge refuses to send orders unless the connected account is a demo account. Do not modify that guard unless you fully understand what you are doing.

## Hardware / stack

- NVIDIA RTX 4050 (Ada, compute 8.9, ~6 GB VRAM)
- Python 3.10–3.12, Windows 11
- CuPy (`cupy-cuda12x`) + Numba CUDA for GPU kernels
- `MetaTrader5` Python package against an Exness MT5 demo terminal

## Project layout

```
src/mcmc_cuda/
  config.py            # paths, constants, env loading
  data/                # historical bar ingestion + parquet cache
  gpu/                 # CuPy/Numba CUDA kernels (MC, Markov chain)
  features/            # returns, regime, strength, slope, momentum
  backtest/            # vectorized engine, costs, metrics
  strategy/            # signal generation combining MC + filters
  broker/              # MT5 bridge (demo-only guard)
  ui/                  # Streamlit backtest UI (Phase 4)
scripts/               # CLI entry points (pull history, run backtest, MT5 sanity)
tests/
notebooks/             # exploration only; outputs are gitignored
data/                  # local cache (gitignored)
```

## Roadmap

- **Phase 1** — Data ingest, CUDA MC paths, Markov chain on discretized returns, baseline vectorized backtester, equity-curve plot.
- **Phase 2** — Regime (HMM/GMM), trend strength (ADX-style), slope, momentum filters; transition matrices conditioned on regime.
- **Phase 3** — ML filter layer (XGBoost first; DL only if XGBoost plateaus).
- **Phase 4** — MT5 live paper-trade bridge + Streamlit backtest UI.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

The MT5 desktop terminal must be installed and logged into a demo account for the broker bridge to work. CuPy will use the CUDA 12.x runtime bundled in its wheel — no separate CUDA toolkit install is required for normal usage.

## License

Proprietary — personal research project.
