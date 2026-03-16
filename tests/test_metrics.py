"""Tests for the metrics utility module."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.metrics import evaluate


class TestEvaluate:
    def test_perfect_predictions(self):
        y = np.array([0, 1, 0, 1, 1])
        metrics = evaluate(y, y, y_prob=y.astype(float))
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["f1"] == pytest.approx(1.0)
        assert metrics["roc_auc"] == pytest.approx(1.0)

    def test_no_prob_gives_none_roc_auc(self):
        y = np.array([0, 1, 0, 1])
        pred = np.array([0, 1, 0, 1])
        metrics = evaluate(y, pred, y_prob=None)
        assert metrics["roc_auc"] is None

    def test_keys_present(self):
        y = np.array([0, 1, 1, 0, 1])
        pred = np.array([0, 1, 0, 0, 1])
        metrics = evaluate(y, pred)
        assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1", "roc_auc"}

    def test_values_in_range(self):
        rng = np.random.default_rng(7)
        y = rng.integers(0, 2, 100)
        pred = rng.integers(0, 2, 100)
        prob = rng.uniform(0, 1, 100)
        metrics = evaluate(y, pred, y_prob=prob)
        for key, val in metrics.items():
            if val is not None:
                assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1]"
