"""Tests for the sklearn models module."""

from __future__ import annotations

import numpy as np
import os
import tempfile
import pytest

from src.models.sklearn_models import (
    MODEL_REGISTRY,
    load_model,
    save_model,
    train_model,
)


def _make_data(n_train: int = 200, n_val: int = 50, n_features: int = 10):
    rng = np.random.default_rng(1)
    X_train = rng.standard_normal((n_train, n_features))
    y_train = rng.integers(0, 2, n_train)
    X_val = rng.standard_normal((n_val, n_features))
    y_val = rng.integers(0, 2, n_val)
    X_test = rng.standard_normal((20, n_features))
    y_test = rng.integers(0, 2, 20)
    return X_train, y_train, X_val, y_val, X_test, y_test


class TestTrainModel:
    @pytest.mark.parametrize("name", list(MODEL_REGISTRY.keys()))
    def test_predict_shape(self, name):
        """Trained model.predict should return array of length n_test."""
        X_train, y_train, X_val, y_val, X_test, y_test = _make_data()
        model = train_model(name, X_train, y_train, X_val, y_val, tune=False)
        preds = model.predict(X_test)
        assert preds.shape == (len(X_test),)

    @pytest.mark.parametrize("name", list(MODEL_REGISTRY.keys()))
    def test_predict_values_binary(self, name):
        """Predictions should only contain 0 and 1."""
        X_train, y_train, X_val, y_val, X_test, y_test = _make_data()
        model = train_model(name, X_train, y_train, X_val, y_val, tune=False)
        preds = model.predict(X_test)
        assert set(preds).issubset({0, 1})

    def test_unknown_model_raises(self):
        """Requesting an unknown model name should raise ValueError."""
        X_train, y_train, X_val, y_val, *_ = _make_data()
        with pytest.raises(ValueError):
            train_model("nonexistent_model", X_train, y_train, X_val, y_val, tune=False)


class TestModelPersistence:
    def test_save_and_load(self):
        """save_model / load_model round-trip should produce identical predictions."""
        X_train, y_train, X_val, y_val, X_test, _ = _make_data()
        model = train_model("random_forest", X_train, y_train, X_val, y_val, tune=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.joblib")
            save_model(model, path)
            loaded = load_model(path)
        np.testing.assert_array_equal(model.predict(X_test), loaded.predict(X_test))
