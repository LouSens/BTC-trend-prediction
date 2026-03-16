"""Tests for the data preprocessor module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessor import get_feature_columns, prepare_data, time_series_split
from src.features.engineer import build_features


def _make_feature_df(n: int = 300) -> pd.DataFrame:
    """Return a synthetic feature-engineered DataFrame."""
    rng = np.random.default_rng(0)
    close = 30_000 + np.cumsum(rng.normal(0, 500, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    high = np.maximum(close, open_) * (1 + rng.uniform(0, 0.01, n))
    low = np.minimum(close, open_) * (1 - rng.uniform(0, 0.01, n))
    volume = rng.integers(1_000, 10_000, n).astype(float)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    raw = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    return build_features(raw)


class TestTimeSeriesSplit:
    def test_sizes_sum_to_total(self):
        df = _make_feature_df()
        train, val, test = time_series_split(df, 0.70, 0.15)
        assert len(train) + len(val) + len(test) == len(df)

    def test_chronological_order(self):
        df = _make_feature_df()
        train, val, test = time_series_split(df)
        assert train.index.max() < val.index.min()
        assert val.index.max() < test.index.min()

    def test_no_overlap(self):
        df = _make_feature_df()
        train, val, test = time_series_split(df)
        assert len(set(train.index) & set(val.index)) == 0
        assert len(set(val.index) & set(test.index)) == 0


class TestPrepareData:
    def test_keys_present(self):
        df = _make_feature_df()
        data = prepare_data(df)
        expected_keys = {"X_train", "X_val", "X_test", "y_train", "y_val", "y_test", "scaler", "feature_cols"}
        assert expected_keys == set(data.keys())

    def test_shapes_consistent(self):
        df = _make_feature_df()
        data = prepare_data(df)
        n_features = len(data["feature_cols"])
        assert data["X_train"].shape[1] == n_features
        assert data["X_val"].shape[1] == n_features
        assert data["X_test"].shape[1] == n_features
        assert len(data["y_train"]) == data["X_train"].shape[0]
        assert len(data["y_val"]) == data["X_val"].shape[0]
        assert len(data["y_test"]) == data["X_test"].shape[0]

    def test_scaler_fitted_when_scale_true(self):
        df = _make_feature_df()
        data = prepare_data(df, scale=True)
        assert data["scaler"] is not None

    def test_no_scaler_when_scale_false(self):
        df = _make_feature_df()
        data = prepare_data(df, scale=False)
        assert data["scaler"] is None

    def test_target_not_in_feature_cols(self):
        df = _make_feature_df()
        data = prepare_data(df)
        assert "Target" not in data["feature_cols"]
