"""Main BTC trend prediction training pipeline.

Usage
-----
Run with defaults (downloads 5 years of BTC-USD data):

    python train.py

Override data range::

    python train.py --start 2020-01-01 --end 2024-01-01

Skip hyperparameter tuning (faster)::

    python train.py --no-tune

Save the best model::

    python train.py --save-dir models/
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

from src.data.loader import load_btc_data
from src.data.preprocessor import prepare_data
from src.features.engineer import build_features
from src.models.sklearn_models import save_model, train_all_models
from src.utils.metrics import evaluate, print_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ML models to predict BTC daily price direction."
    )
    parser.add_argument("--ticker", default="BTC-USD", help="Yahoo Finance ticker symbol.")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD).")
    parser.add_argument("--period", default=None, help="yfinance period shorthand (e.g. '5y').")
    parser.add_argument(
        "--train-ratio", type=float, default=0.70, help="Fraction of data for training."
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.15, help="Fraction of data for validation."
    )
    parser.add_argument(
        "--no-tune", action="store_true", help="Skip hyperparameter tuning (faster)."
    )
    parser.add_argument(
        "--n-iter", type=int, default=20, help="RandomizedSearchCV iterations when tuning."
    )
    parser.add_argument(
        "--cv", type=int, default=3, help="Cross-validation folds when tuning."
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Directory to save trained models (omit to skip saving).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> dict:
    """Execute the full training pipeline and return evaluation results.

    Returns:
        Dict mapping model name → test metrics dict.
    """
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    logger.info("Step 1/5 — Loading BTC price data.")
    raw_df = load_btc_data(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        period=args.period,
    )
    logger.info("Raw data shape: %s", raw_df.shape)

    # ------------------------------------------------------------------
    # 2. Feature engineering
    # ------------------------------------------------------------------
    logger.info("Step 2/5 — Engineering features.")
    feat_df = build_features(raw_df)
    logger.info("Feature matrix shape: %s", feat_df.shape)
    logger.info(
        "Class balance — Up: %.1f%%, Down: %.1f%%",
        feat_df["Target"].mean() * 100,
        (1 - feat_df["Target"].mean()) * 100,
    )

    # ------------------------------------------------------------------
    # 3. Prepare train / val / test splits
    # ------------------------------------------------------------------
    logger.info("Step 3/5 — Preparing data splits.")
    data = prepare_data(
        feat_df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        scale=True,
    )

    # ------------------------------------------------------------------
    # 4. Train models
    # ------------------------------------------------------------------
    logger.info("Step 4/5 — Training models (tune=%s).", not args.no_tune)
    trained_models = train_all_models(
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
        tune=not args.no_tune,
        n_iter=args.n_iter,
        cv=args.cv,
    )

    # ------------------------------------------------------------------
    # 5. Evaluate on held-out test set
    # ------------------------------------------------------------------
    logger.info("Step 5/5 — Evaluating on test set.")
    results: dict[str, dict] = {}

    for name, model in trained_models.items():
        y_pred = model.predict(data["X_test"])
        y_prob = (
            model.predict_proba(data["X_test"])[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )
        metrics = evaluate(data["y_test"], y_pred, y_prob)
        results[name] = metrics
        print_metrics(name, metrics)

    # ------------------------------------------------------------------
    # Select and optionally save the best model (by F1 score)
    # ------------------------------------------------------------------
    best_name = max(results, key=lambda n: results[n].get("f1") or 0.0)
    logger.info("Best model by F1: %s (F1=%.4f)", best_name, results[best_name]["f1"])

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        for name, model in trained_models.items():
            path = os.path.join(args.save_dir, f"{name}.joblib")
            save_model(model, path)

        # Also save the scaler.
        if data["scaler"] is not None:
            import joblib

            scaler_path = os.path.join(args.save_dir, "scaler.joblib")
            joblib.dump(data["scaler"], scaler_path)
            logger.info("Scaler saved to %s", scaler_path)

    return results


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    results = run_pipeline(args)
    sys.exit(0)
