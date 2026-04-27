"""1st-order Markov chain on discretized log-returns, with GPU path sampling.

Pipeline:
1. discretize_returns(): bin a series of log-returns into K states using
   equal-quantile boundaries computed on the same series (rolling-quantile
   adaptation is added in Phase 2 once regimes are explicit).
2. transition_matrix(): empirical K x K transition counts -> row-normalized
   probabilities, with Laplace smoothing so zero-count transitions don't
   trap the sampler.
3. sample_paths(): GPU-batch sample n_paths each of length `horizon` from
   a starting state, returning state sequences. We use a Numba CUDA kernel
   because CuPy's prefix-sum-then-binary-search trick per step is awkward
   to vectorize cleanly across heterogeneous starting states.

The strategy layer combines the resulting state distribution with a
state-to-expected-return map (computed at fit time) to produce a directional
forecast comparable to the MC simulator's output.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mcmc_cuda.gpu.device import get_xp, has_cuda


@dataclass
class MarkovModel:
    """Fitted 1st-order Markov chain over discretized log-returns."""

    n_states: int
    bin_edges: np.ndarray              # shape (n_states + 1,)
    transition: np.ndarray             # (K, K) row-stochastic
    state_mean_return: np.ndarray      # (K,) avg log-return per state
    state_count: np.ndarray            # (K,) sample counts per state


def discretize_returns(log_returns: np.ndarray, n_states: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Equal-quantile binning. Returns (state_indices, bin_edges)."""
    if log_returns.ndim != 1:
        raise ValueError("log_returns must be 1D")
    qs = np.linspace(0.0, 1.0, n_states + 1)
    edges = np.quantile(log_returns, qs)
    # Ensure strictly increasing edges (degenerate when many duplicate returns).
    edges = np.maximum.accumulate(edges + 1e-12 * np.arange(edges.size))
    edges[0], edges[-1] = -np.inf, np.inf
    states = np.searchsorted(edges, log_returns, side="right") - 1
    states = np.clip(states, 0, n_states - 1)
    return states.astype(np.int32), edges


def fit_markov(log_returns: np.ndarray, n_states: int = 5, smoothing: float = 1.0) -> MarkovModel:
    """Fit a 1st-order Markov chain on discretized log-returns."""
    states, edges = discretize_returns(log_returns, n_states)
    counts = np.zeros((n_states, n_states), dtype=np.float64)
    np.add.at(counts, (states[:-1], states[1:]), 1.0)
    counts += smoothing
    trans = counts / counts.sum(axis=1, keepdims=True)

    state_mean = np.zeros(n_states, dtype=np.float64)
    state_count = np.zeros(n_states, dtype=np.int64)
    for s in range(n_states):
        mask = states == s
        state_count[s] = int(mask.sum())
        state_mean[s] = float(log_returns[mask].mean()) if state_count[s] > 0 else 0.0

    return MarkovModel(
        n_states=n_states,
        bin_edges=edges,
        transition=trans,
        state_mean_return=state_mean,
        state_count=state_count,
    )


def state_of(return_value: float, model: MarkovModel) -> int:
    s = int(np.searchsorted(model.bin_edges, return_value, side="right") - 1)
    return max(0, min(model.n_states - 1, s))


# ---------- GPU sampler --------------------------------------------------


def _build_cuda_kernel():
    from numba import cuda

    @cuda.jit
    def _kernel(start_state, cum_trans, n_states, horizon, randoms, out):
        i = cuda.grid(1)
        if i >= out.shape[0]:
            return
        s = start_state
        for t in range(horizon):
            r = randoms[i, t]
            # Linear scan over the row's CDF (K small, branch-predictable).
            row_off = s * n_states
            picked = n_states - 1
            for k in range(n_states):
                if r <= cum_trans[row_off + k]:
                    picked = k
                    break
            s = picked
            out[i, t] = s

    return _kernel


_CUDA_KERNEL = None


def sample_paths(
    model: MarkovModel,
    start_state: int,
    horizon: int,
    n_paths: int = 50_000,
    seed: int | None = None,
) -> np.ndarray:
    """Sample n_paths state sequences of length `horizon` starting from start_state.

    Returns int32 ndarray of shape (n_paths, horizon).
    """
    if not 0 <= start_state < model.n_states:
        raise ValueError(f"start_state {start_state} out of range [0, {model.n_states})")

    cum = np.cumsum(model.transition, axis=1).astype(np.float32).reshape(-1)

    if has_cuda():
        return _sample_gpu(cum, model.n_states, start_state, horizon, n_paths, seed)
    return _sample_cpu(cum, model.n_states, start_state, horizon, n_paths, seed)


def _sample_gpu(cum_flat, n_states, start_state, horizon, n_paths, seed):
    import cupy as cp
    from numba import cuda

    global _CUDA_KERNEL
    if _CUDA_KERNEL is None:
        _CUDA_KERNEL = _build_cuda_kernel()

    rng = cp.random.default_rng(seed)
    randoms = rng.random((n_paths, horizon), dtype=cp.float32)
    out = cp.empty((n_paths, horizon), dtype=cp.int32)
    cum_d = cp.asarray(cum_flat)

    threads = 256
    blocks = (n_paths + threads - 1) // threads
    # Numba CUDA accepts CuPy arrays via __cuda_array_interface__.
    _CUDA_KERNEL[blocks, threads](
        np.int32(start_state), cum_d, np.int32(n_states),
        np.int32(horizon), randoms, out,
    )
    cuda.synchronize()
    return cp.asnumpy(out)


def _sample_cpu(cum_flat, n_states, start_state, horizon, n_paths, seed):
    rng = np.random.default_rng(seed)
    randoms = rng.random((n_paths, horizon)).astype(np.float32)
    out = np.empty((n_paths, horizon), dtype=np.int32)
    for i in range(n_paths):
        s = start_state
        for t in range(horizon):
            row = cum_flat[s * n_states:(s + 1) * n_states]
            s = int(np.searchsorted(row, randoms[i, t]))
            s = min(s, n_states - 1)
            out[i, t] = s
    return out


def forecast_from_paths(
    state_paths: np.ndarray, model: MarkovModel
) -> tuple[float, float, np.ndarray]:
    """Aggregate state sequences into (prob_up, expected_log_return, terminal_state_dist).

    Sum of state_mean_return across the path approximates the path's log-return,
    which is exact in the limit of fine state binning.
    """
    means = model.state_mean_return[state_paths]   # (n_paths, horizon)
    horizon_log_r = means.sum(axis=1)
    prob_up = float((horizon_log_r > 0).mean())
    expected_log_return = float(horizon_log_r.mean())
    terminal = state_paths[:, -1]
    dist = np.bincount(terminal, minlength=model.n_states) / terminal.size
    return prob_up, expected_log_return, dist
