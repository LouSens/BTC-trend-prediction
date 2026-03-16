"""Evaluation metrics for binary classification models."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute a standard set of binary classification metrics.

    Args:
        y_true: Ground-truth labels (0 or 1).
        y_pred: Predicted labels (0 or 1).
        y_prob: Predicted probabilities for the positive class.  Required for
                ROC-AUC; if ``None`` the ROC-AUC entry will be ``None``.

    Returns:
        Dict with ``accuracy``, ``precision``, ``recall``, ``f1``,
        and ``roc_auc`` (or ``None``).
    """
    metrics: dict[str, float | None] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if y_prob is not None else None,
    }
    return metrics


def print_metrics(name: str, metrics: dict[str, float | None]) -> None:
    """Pretty-print a metrics dictionary."""
    print(f"\n{'=' * 50}")
    print(f"  Model: {name}")
    print(f"{'=' * 50}")
    for key, value in metrics.items():
        if value is None:
            print(f"  {key:<12}: N/A")
        else:
            print(f"  {key:<12}: {value:.4f}")
    print(f"{'=' * 50}\n")
