"""Ensemble signal generator combining MC + Markov forecasts into one direction.

Phase 1 ensemble (intentionally simple, edge comes in Phase 2 with filters):
- Fit a Markov chain on a rolling training window of past log-returns.
- For each bar, run BOTH a bootstrap MC and a Markov-chain forecast over
  the same horizon.
- Average prob_up across the two models. If avg prob_up exceeds
  prob_threshold, signal = +1 (long); if below 1 - prob_threshold,
  signal = -1 (short); else 0 (flat).
- Also require expected_log_return to agree in sign with the directional
  call so a knife-edge probability with adverse expectation doesn't trade.

This produces a per-bar signal column that the backtester will consume.
The rolling refit is expensive — we expose `refit_every` so users can refit
e.g. once per 96 bars (1 day at M15) instead of every bar.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mcmc_cuda.gpu.markov import fit_markov, forecast_from_paths, sample_paths, state_of
from mcmc_cuda.gpu.monte_carlo import bootstrap_paths


@dataclass
class EnsembleConfig:
    horizon: int = 16             # bars; default 16 ~= 4h on M15
    train_window: int = 2000      # bars used to fit Markov + bootstrap pool
    n_states: int = 5
    n_mc_paths: int = 50_000
    n_markov_paths: int = 50_000
    prob_threshold: float = 0.55  # required avg prob_up for long (and 1-x for short)
    refit_every: int = 96         # refit Markov chain every N bars (M15: ~1 day)
    seed: int | None = 42


def generate_signals(close: pd.Series, cfg: EnsembleConfig | None = None) -> pd.DataFrame:
    """Compute per-bar directional signal and forecast diagnostics.

    Returns a DataFrame indexed like `close` with columns:
        signal (-1/0/+1), prob_up_mc, prob_up_markov, prob_up_avg,
        exp_logret_mc, exp_logret_markov, current_state
    """
    cfg = cfg or EnsembleConfig()
    log_ret = np.log(close).diff().dropna().values.astype(np.float64)
    idx = close.index[1:]  # log_ret aligned to bar t = ret from t-1 to t

    n = log_ret.size
    out = np.full((n, 7), np.nan, dtype=np.float64)
    model = None
    last_fit = -10**9

    for i in range(cfg.train_window, n):
        if i - last_fit >= cfg.refit_every or model is None:
            window = log_ret[i - cfg.train_window:i]
            model = fit_markov(window, n_states=cfg.n_states)
            last_fit = i

        current_ret = log_ret[i]
        s0 = state_of(current_ret, model)

        mc = bootstrap_paths(
            log_ret[i - cfg.train_window:i],
            horizon=cfg.horizon,
            n_paths=cfg.n_mc_paths,
            seed=cfg.seed,
        )
        paths = sample_paths(
            model, start_state=s0, horizon=cfg.horizon,
            n_paths=cfg.n_markov_paths, seed=cfg.seed,
        )
        p_mk, e_mk, _ = forecast_from_paths(paths, model)

        prob_avg = 0.5 * (mc.prob_up + p_mk)
        e_avg = 0.5 * (mc.expected_log_return + e_mk)

        sig = 0
        if prob_avg >= cfg.prob_threshold and e_avg > 0:
            sig = 1
        elif prob_avg <= 1 - cfg.prob_threshold and e_avg < 0:
            sig = -1

        out[i] = [sig, mc.prob_up, p_mk, prob_avg, mc.expected_log_return, e_mk, s0]

    df = pd.DataFrame(
        out,
        index=idx,
        columns=[
            "signal", "prob_up_mc", "prob_up_markov", "prob_up_avg",
            "exp_logret_mc", "exp_logret_markov", "current_state",
        ],
    )
    df["signal"] = df["signal"].fillna(0).astype(int)
    return df
