"""Monte Carlo path simulator on GPU (CuPy).

Two simulators:
- `gbm_paths`: geometric Brownian motion baseline (drift mu, vol sigma).
  Closed-form vectorized — fast, useful as a sanity baseline.
- `bootstrap_paths`: empirical bootstrap of historical log-returns —
  no distributional assumption, captures fat tails and serial-correlation-
  free properties of the realized return distribution.

Both return arrays of shape (n_paths, horizon) of *log-returns* (not prices).
The strategy layer aggregates these into directional probability and
expected-return forecasts.

VRAM accounting: float32 paths cost 4 bytes per cell. On the RTX 4050
(~6 GB), a (n_paths=200_000, horizon=64) buffer is ~50 MB — fine. If the
caller asks for something that exceeds free VRAM, we chunk.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mcmc_cuda.gpu.device import free_vram_bytes, get_xp, has_cuda


@dataclass
class MCResult:
    """Aggregate forecasts from a path simulation."""

    prob_up: float           # P(return at horizon > 0)
    expected_log_return: float
    p05: float               # 5th percentile of horizon log-return
    p95: float
    n_paths: int


def _chunk_size(n_paths: int, horizon: int, dtype_bytes: int = 4) -> int:
    """Pick a chunk size that fits in ~25% of free VRAM (leave room for other buffers)."""
    free = free_vram_bytes()
    if free is None:
        return n_paths
    budget = int(free * 0.25)
    per_path = horizon * dtype_bytes
    return max(1024, min(n_paths, budget // max(per_path, 1)))


def gbm_paths(
    mu: float,
    sigma: float,
    horizon: int,
    n_paths: int = 100_000,
    seed: int | None = None,
) -> MCResult:
    """Simulate horizon-step GBM log-returns and return aggregate stats.

    mu, sigma are *per-step* (already scaled to the bar timeframe).
    """
    xp = get_xp()
    rng = xp.random.default_rng(seed)
    chunk = _chunk_size(n_paths, horizon)

    sum_up = 0
    sum_logr = 0.0
    samples_q = []  # collect per-chunk horizon log-returns for quantiles

    remaining = n_paths
    while remaining > 0:
        m = min(chunk, remaining)
        # Antithetic variates: pair each draw with its negation to halve variance.
        half = m // 2
        z_half = rng.standard_normal((half, horizon), dtype=xp.float32)
        z = xp.concatenate([z_half, -z_half], axis=0)
        if z.shape[0] < m:  # odd m: top up with one extra row
            extra = rng.standard_normal((m - z.shape[0], horizon), dtype=xp.float32)
            z = xp.concatenate([z, extra], axis=0)

        log_r_step = (mu - 0.5 * sigma * sigma) + sigma * z
        horizon_log_r = log_r_step.sum(axis=1)
        sum_up += int((horizon_log_r > 0).sum().get() if has_cuda() else (horizon_log_r > 0).sum())
        sum_logr += float(horizon_log_r.sum().get() if has_cuda() else horizon_log_r.sum())
        samples_q.append(_to_numpy(horizon_log_r))
        remaining -= m

    flat = np.concatenate(samples_q)
    return MCResult(
        prob_up=sum_up / n_paths,
        expected_log_return=sum_logr / n_paths,
        p05=float(np.quantile(flat, 0.05)),
        p95=float(np.quantile(flat, 0.95)),
        n_paths=n_paths,
    )


def bootstrap_paths(
    historical_log_returns: np.ndarray,
    horizon: int,
    n_paths: int = 100_000,
    seed: int | None = None,
) -> MCResult:
    """Sample horizon-step paths by bootstrapping from `historical_log_returns`.

    Each path is a sum of `horizon` iid draws from the empirical distribution.
    """
    xp = get_xp()
    if historical_log_returns.size < 32:
        raise ValueError("Need at least 32 historical returns to bootstrap.")

    pool = xp.asarray(historical_log_returns, dtype=xp.float32)
    n_pool = pool.shape[0]
    rng = xp.random.default_rng(seed)
    chunk = _chunk_size(n_paths, horizon)

    sum_up = 0
    sum_logr = 0.0
    samples_q = []

    remaining = n_paths
    while remaining > 0:
        m = min(chunk, remaining)
        idx = rng.integers(0, n_pool, size=(m, horizon), dtype=xp.int64)
        draws = pool[idx]
        horizon_log_r = draws.sum(axis=1)
        sum_up += int((horizon_log_r > 0).sum().get() if has_cuda() else (horizon_log_r > 0).sum())
        sum_logr += float(horizon_log_r.sum().get() if has_cuda() else horizon_log_r.sum())
        samples_q.append(_to_numpy(horizon_log_r))
        remaining -= m

    flat = np.concatenate(samples_q)
    return MCResult(
        prob_up=sum_up / n_paths,
        expected_log_return=sum_logr / n_paths,
        p05=float(np.quantile(flat, 0.05)),
        p95=float(np.quantile(flat, 0.95)),
        n_paths=n_paths,
    )


def _to_numpy(arr) -> np.ndarray:
    return arr.get() if has_cuda() else np.asarray(arr)
