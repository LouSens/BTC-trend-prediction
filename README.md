# BTC Trend Prediction

A machine learning pipeline that predicts the **daily price direction** (up or down) of Bitcoin using technical indicators and scikit-learn classifiers.

---

## Overview

The pipeline:

1. **Downloads** historical BTC-USD OHLCV data from Yahoo Finance via `yfinance`.
2. **Engineers features** — SMA, EMA, RSI, MACD, Bollinger Bands, ATR, volume ratio, price returns, and lag returns.
3. **Splits** data chronologically into train / validation / test sets (no look-ahead bias).
4. **Trains** three classifiers with optional hyperparameter tuning:
   - Random Forest
   - Gradient Boosting
   - Logistic Regression
5. **Evaluates** each model on the held-out test set (accuracy, precision, recall, F1, ROC-AUC).
6. **Saves** trained models and the fitted scaler to disk (optional).

---

## Project Structure

```
BTC-trend-prediction/
├── requirements.txt          # Python dependencies
├── train.py                  # Main pipeline entrypoint
├── src/
│   ├── data/
│   │   ├── loader.py         # BTC data download (yfinance)
│   │   └── preprocessor.py   # Train/val/test split & scaling
│   ├── features/
│   │   └── engineer.py       # Technical indicator computation
│   ├── models/
│   │   └── sklearn_models.py # Model definitions, training, persistence
│   └── utils/
│       └── metrics.py        # Classification metric helpers
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_preprocessor.py
│   ├── test_models.py
│   └── test_metrics.py
└── models/                   # Saved models (created at runtime)
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### Run with defaults (5 years of BTC-USD, with hyperparameter tuning)

```bash
python train.py
```

### Custom date range

```bash
python train.py --start 2020-01-01 --end 2024-01-01
```

### Skip hyperparameter tuning (faster)

```bash
python train.py --no-tune
```

### Save trained models

```bash
python train.py --save-dir models/
```

### All options

```
usage: train.py [-h] [--ticker TICKER] [--start START] [--end END]
                [--period PERIOD] [--train-ratio TRAIN_RATIO]
                [--val-ratio VAL_RATIO] [--no-tune] [--n-iter N_ITER]
                [--cv CV] [--save-dir SAVE_DIR]
```

---

## Features Engineered

| Category       | Features                                               |
|---------------|--------------------------------------------------------|
| Trend          | SMA(10/20/50), EMA(10/20/50)                           |
| Momentum       | RSI(14), MACD, MACD Signal, MACD Histogram             |
| Volatility     | Bollinger Bands (upper/middle/lower/%B), ATR(14)       |
| Volume         | Volume SMA(20), Volume Ratio                           |
| Price-derived  | Daily Return, Log Return, HL Ratio, OC Ratio           |
| Lag features   | 1/2/3/5/10-day lagged returns                          |

**Target:** Binary label — `1` if next-day close > current close, `0` otherwise.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

> **Note:** `tests/test_data.py` requires an active internet connection to download BTC data from Yahoo Finance.
