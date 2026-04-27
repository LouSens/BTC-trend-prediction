"""Markov chain unit tests. Use MCMC_FORCE_CPU=1 path so CI without GPU works."""
from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("MCMC_FORCE_CPU", "1")

from mcmc_cuda.gpu.markov import (
    discretize_returns,
    fit_markov,
    forecast_from_paths,
    sample_paths,
    state_of,
)


def test_discretize_returns_balanced():
    rng = np.random.default_rng(0)
    r = rng.standard_normal(10_000)
    states, edges = discretize_returns(r, n_states=5)
    assert states.dtype == np.int32
    assert states.min() == 0 and states.max() == 4
    counts = np.bincount(states, minlength=5)
    # equal-quantile binning should produce ~equal counts
    assert counts.max() / counts.min() < 1.1


def test_fit_markov_row_stochastic():
    rng = np.random.default_rng(1)
    r = rng.standard_normal(5_000) * 0.001
    m = fit_markov(r, n_states=4)
    assert m.transition.shape == (4, 4)
    assert np.allclose(m.transition.sum(axis=1), 1.0)


def test_state_of_within_bounds():
    rng = np.random.default_rng(2)
    r = rng.standard_normal(2_000) * 0.001
    m = fit_markov(r, n_states=5)
    for v in (-10.0, 0.0, 10.0, r[100]):
        s = state_of(v, m)
        assert 0 <= s < m.n_states


def test_sample_and_forecast_reasonable():
    rng = np.random.default_rng(3)
    r = rng.normal(loc=0.0, scale=0.001, size=4_000)
    m = fit_markov(r, n_states=5)
    paths = sample_paths(m, start_state=2, horizon=8, n_paths=5_000, seed=7)
    assert paths.shape == (5_000, 8)
    p_up, e_logr, dist = forecast_from_paths(paths, m)
    assert 0.3 < p_up < 0.7   # symmetric noise -> probability near 0.5
    assert dist.shape == (5,) and abs(dist.sum() - 1.0) < 1e-9
