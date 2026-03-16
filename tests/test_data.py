"""Tests for the data loader module."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.loader import load_btc_data


class TestLoadBtcData:
    """Tests for load_btc_data using a small offline yfinance download."""

    def test_returns_dataframe(self):
        """load_btc_data should return a non-empty DataFrame."""
        df = load_btc_data(period="1mo")
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_has_required_columns(self):
        """Returned DataFrame should contain standard OHLCV columns."""
        df = load_btc_data(period="1mo")
        required = {"Open", "High", "Low", "Close", "Volume"}
        assert required.issubset(set(df.columns)), (
            f"Missing columns: {required - set(df.columns)}"
        )

    def test_index_is_datetime(self):
        """The DataFrame index should be a DatetimeIndex."""
        df = load_btc_data(period="1mo")
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_no_all_nan_rows(self):
        """There should be no rows where every value is NaN."""
        df = load_btc_data(period="1mo")
        assert not df.isnull().all(axis=1).any()

    def test_start_end_dates(self):
        """Providing start/end dates should restrict the returned data."""
        df = load_btc_data(start="2022-01-01", end="2022-12-31")
        assert not df.empty
        assert df.index.min().year == 2022
        assert df.index.max().year == 2022

    def test_invalid_ticker_raises(self):
        """An invalid ticker should raise a ValueError."""
        with pytest.raises(ValueError):
            load_btc_data(ticker="INVALID_TICKER_XYZ_123", period="1mo")
