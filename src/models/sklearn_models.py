"""Scikit-learn based classifiers for BTC trend prediction."""

from __future__ import annotations

import logging
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------

def _rf_search_space() -> dict[str, Any]:
    return {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 15],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }


def _gb_search_space() -> dict[str, Any]:
    return {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "min_samples_split": [2, 5],
    }


def _lr_search_space() -> dict[str, Any]:
    return {
        "C": [0.01, 0.1, 1.0, 10.0],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"],
        "max_iter": [500],
    }


MODEL_REGISTRY: dict[str, tuple[Any, dict[str, Any]]] = {
    "random_forest": (
        RandomForestClassifier(random_state=42, n_jobs=-1),
        _rf_search_space(),
    ),
    "gradient_boosting": (
        GradientBoostingClassifier(random_state=42),
        _gb_search_space(),
    ),
    "logistic_regression": (
        LogisticRegression(random_state=42),
        _lr_search_space(),
    ),
}


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_model(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    tune: bool = True,
    n_iter: int = 20,
    cv: int = 3,
    random_state: int = 42,
) -> Any:
    """Train a single named model, optionally with randomised hyperparameter search.

    Args:
        name:         Key in :data:`MODEL_REGISTRY`.
        X_train:      Training feature matrix.
        y_train:      Training labels.
        X_val:        Validation feature matrix (concatenated with train for CV
                      when ``tune=True``).
        y_val:        Validation labels.
        tune:         If ``True``, run ``RandomizedSearchCV`` to tune
                      hyperparameters.
        n_iter:       Number of random parameter combinations to try.
        cv:           Cross-validation folds (applied on train set only).
        random_state: Random seed for reproducibility.

    Returns:
        Fitted estimator (best estimator when ``tune=True``).
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")

    estimator, search_space = MODEL_REGISTRY[name]
    if hasattr(estimator, "random_state"):
        estimator.set_params(random_state=random_state)

    # Combine train + val for hyperparameter search to use more data.
    X_fit = np.vstack([X_train, X_val])
    y_fit = np.concatenate([y_train, y_val])

    if tune and search_space:
        logger.info("Tuning %s with RandomizedSearchCV (n_iter=%d, cv=%d).", name, n_iter, cv)
        search = RandomizedSearchCV(
            estimator,
            param_distributions=search_space,
            n_iter=n_iter,
            cv=cv,
            scoring="f1",
            n_jobs=-1,
            random_state=random_state,
            refit=True,
        )
        search.fit(X_fit, y_fit)
        best = search.best_estimator_
        logger.info("Best params for %s: %s", name, search.best_params_)
    else:
        logger.info("Training %s without hyperparameter search.", name)
        best = estimator
        best.fit(X_fit, y_fit)

    return best


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    tune: bool = True,
    n_iter: int = 20,
    cv: int = 3,
) -> dict[str, Any]:
    """Train all models in :data:`MODEL_REGISTRY` and return a name→model dict."""
    trained: dict[str, Any] = {}
    for name in MODEL_REGISTRY:
        logger.info("Training model: %s", name)
        trained[name] = train_model(
            name,
            X_train,
            y_train,
            X_val,
            y_val,
            tune=tune,
            n_iter=n_iter,
            cv=cv,
        )
    return trained


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_model(model: Any, path: str) -> None:
    """Persist *model* to disk using :mod:`joblib`."""
    joblib.dump(model, path)
    logger.info("Model saved to %s", path)


def load_model(path: str) -> Any:
    """Load and return a model from *path* using :mod:`joblib`."""
    model = joblib.load(path)
    logger.info("Model loaded from %s", path)
    return model
