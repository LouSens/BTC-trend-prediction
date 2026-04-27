"""Monte Carlo simulator unit tests."""
from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("MCMC_FORCE_CPU", "1")

from mcmc_cuda.gpu.monte_carlo import bootstrap_paths, gbm_paths


def test_gbm_zero_drift_symmetric():
    res = gbm_paths(mu=0.0, sigma=0.01, horizon=20, n_paths=20_000, seed=42)
    assert 0.45 < res.prob_up < 0.55
    assert abs(res.expected_log_return) < 0.005


def test_gbm_positive_drift_biased_up():
    res = gbm_paths(mu=0.001, sigma=0.005, horizon=40, n_paths=20_000, seed=42)
    assert res.prob_up > 0.6
    assert res.expected_log_return > 0


def test_bootstrap_matches_input_mean():
    rng = np.random.default_rng(0)
    sample = rng.normal(0.0, 0.001, size=2_000)
    res = bootstrap_paths(sample, horizon=10, n_paths=10_000, seed=1)
    # mean of horizon log-returns should ~ horizon * sample mean (~ 0)
    assert abs(res.expected_log_return) < 0.003
