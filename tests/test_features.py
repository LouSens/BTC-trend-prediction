"""Tests for the feature engineering module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import (
    add_atr,
    add_bollinger_bands,
    add_ema,
    add_lag_features,
    add_macd,
    add_price_features,
    add_rsi,
    add_sma,
    add_volume_features,
    build_features,
    create_target,
)


def _make_ohlcv(n: int = 200) -> pd.DataFrame:
    """Return a synthetic OHLCV DataFrame with *n* rows."""
    rng = np.random.default_rng(42)
    close = 30_000 + np.cumsum(rng.normal(0, 500, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    high = np.maximum(close, open_) * (1 + rng.uniform(0, 0.01, n))
    low = np.minimum(close, open_) * (1 - rng.uniform(0, 0.01, n))
    volume = rng.integers(1_000, 10_000, n).astype(float)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


class TestIndividualIndicators:
    def test_sma_columns_added(self):
        df = _make_ohlcv()
        result = add_sma(df.copy(), windows=[10, 20])
        assert "SMA_10" in result.columns
        assert "SMA_20" in result.columns

    def test_ema_columns_added(self):
        df = _make_ohlcv()
        result = add_ema(df.copy(), windows=[10])
        assert "EMA_10" in result.columns

    def test_rsi_range(self):
        df = _make_ohlcv(100)
        result = add_rsi(df.copy())
        rsi = result["RSI_14"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_macd_columns_added(self):
        df = _make_ohlcv()
        result = add_macd(df.copy())
        assert {"MACD", "MACD_Signal", "MACD_Hist"}.issubset(result.columns)

    def test_bollinger_bands_columns_added(self):
        df = _make_ohlcv()
        result = add_bollinger_bands(df.copy())
        assert {"BB_Upper", "BB_Middle", "BB_Lower", "BB_PctB"}.issubset(result.columns)

    def test_atr_positive(self):
        df = _make_ohlcv(100)
        result = add_atr(df.copy())
        atr = result["ATR_14"].dropna()
        assert (atr >= 0).all()

    def test_volume_features(self):
        df = _make_ohlcv()
        result = add_volume_features(df.copy())
        assert "Volume_SMA" in result.columns
        assert "Volume_Ratio" in result.columns

    def test_price_features(self):
        df = _make_ohlcv()
        result = add_price_features(df.copy())
        assert {"Daily_Return", "Log_Return", "Daily_Range_Pct", "OC_Ratio"}.issubset(result.columns)

    def test_lag_features(self):
        df = _make_ohlcv()
        result = add_lag_features(df.copy(), lags=[1, 3])
        assert "Return_Lag_1" in result.columns
        assert "Return_Lag_3" in result.columns

    def test_target_binary(self):
        df = _make_ohlcv()
        result = create_target(df.copy())
        unique = set(result["Target"].dropna().unique())
        assert unique.issubset({0, 1})


class TestBuildFeatures:
    def test_no_nan_after_build(self):
        df = _make_ohlcv(300)
        result = build_features(df)
        assert not result.isnull().any().any(), "build_features should return NaN-free DataFrame"

    def test_has_target_column(self):
        df = _make_ohlcv(300)
        result = build_features(df)
        assert "Target" in result.columns

    def test_fewer_rows_than_input(self):
        """Rolling windows and lags should reduce the number of rows."""
        df = _make_ohlcv(300)
        result = build_features(df)
        assert len(result) < len(df)

    def test_target_is_binary(self):
        df = _make_ohlcv(300)
        result = build_features(df)
        assert set(result["Target"].unique()).issubset({0, 1})
