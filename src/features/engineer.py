"""Feature engineering: technical indicators for BTC price data."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual indicator helpers
# ---------------------------------------------------------------------------

def add_sma(df: pd.DataFrame, column: str = "Close", windows: list[int] | None = None) -> pd.DataFrame:
    """Add Simple Moving Averages for each window size."""
    if windows is None:
        windows = [10, 20, 50]
    for w in windows:
        df[f"SMA_{w}"] = df[column].rolling(window=w).mean()
    return df


def add_ema(df: pd.DataFrame, column: str = "Close", windows: list[int] | None = None) -> pd.DataFrame:
    """Add Exponential Moving Averages for each window size."""
    if windows is None:
        windows = [10, 20, 50]
    for w in windows:
        df[f"EMA_{w}"] = df[column].ewm(span=w, adjust=False).mean()
    return df


def add_rsi(df: pd.DataFrame, column: str = "Close", period: int = 14) -> pd.DataFrame:
    """Add Relative Strength Index (RSI).

    RSI = 100 - 100 / (1 + RS), where RS = avg_gain / avg_loss.
    """
    delta = df[column].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df[f"RSI_{period}"] = 100 - (100 / (1 + rs))
    return df


def add_macd(
    df: pd.DataFrame,
    column: str = "Close",
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Add MACD, MACD signal line, and MACD histogram."""
    ema_fast = df[column].ewm(span=fast, adjust=False).mean()
    ema_slow = df[column].ewm(span=slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    return df


def add_bollinger_bands(
    df: pd.DataFrame,
    column: str = "Close",
    window: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """Add Bollinger Bands (upper, middle, lower) and %B indicator."""
    rolling_mean = df[column].rolling(window=window).mean()
    rolling_std = df[column].rolling(window=window).std()
    df["BB_Upper"] = rolling_mean + num_std * rolling_std
    df["BB_Middle"] = rolling_mean
    df["BB_Lower"] = rolling_mean - num_std * rolling_std
    band_width = df["BB_Upper"] - df["BB_Lower"]
    df["BB_PctB"] = (df[column] - df["BB_Lower"]) / band_width.replace(0, np.nan)
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Average True Range (ATR) as a volatility measure."""
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df[f"ATR_{period}"] = true_range.ewm(com=period - 1, adjust=False).mean()
    return df


def add_volume_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Add volume-based features: rolling average and relative volume."""
    df["Volume_SMA"] = df["Volume"].rolling(window=window).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA"].replace(0, np.nan)
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add price-derived features: daily return, log return, and HL ratio."""
    df["Daily_Return"] = df["Close"].pct_change()
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Daily_Range_Pct"] = (df["High"] - df["Low"]) / df["Close"].shift(1).replace(0, np.nan)
    df["OC_Ratio"] = (df["Close"] - df["Open"]) / df["Open"].replace(0, np.nan)
    return df


def add_lag_features(df: pd.DataFrame, column: str = "Close", lags: list[int] | None = None) -> pd.DataFrame:
    """Add lagged returns as additional predictive features."""
    if lags is None:
        lags = [1, 2, 3, 5, 10]
    for lag in lags:
        df[f"Return_Lag_{lag}"] = df[column].pct_change().shift(lag)
    return df


def create_target(df: pd.DataFrame, column: str = "Close", forward: int = 1) -> pd.DataFrame:
    """Create binary target: 1 if next *forward*-day return is positive, else 0."""
    future_return = df[column].shift(-forward) / df[column] - 1
    df["Target"] = (future_return > 0).astype(int)
    return df


# ---------------------------------------------------------------------------
# Composite pipeline
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps and return a clean DataFrame.

    The function modifies a *copy* of the input to avoid mutating the original.
    Rows containing NaN values (introduced by rolling windows and lag
    operations) are dropped before returning.

    Args:
        df: Raw OHLCV DataFrame with at least ``Close``, ``High``, ``Low``,
            ``Open``, and ``Volume`` columns.

    Returns:
        Feature-enriched DataFrame with a ``Target`` column and no NaN rows.
    """
    df = df.copy()

    logger.info("Engineering features from %d raw rows.", len(df))

    df = add_sma(df)
    df = add_ema(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_atr(df)
    df = add_volume_features(df)
    df = add_price_features(df)
    df = add_lag_features(df)
    df = create_target(df)

    initial_len = len(df)
    df = df.dropna()
    logger.info(
        "Feature engineering complete: %d rows retained (dropped %d NaN rows).",
        len(df),
        initial_len - len(df),
    )
    return df
