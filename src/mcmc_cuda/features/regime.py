"""Regime detection: 2-state Gaussian HMM on (return, abs-return) features.

Why HMM here:
- Markets switch between trending (drift dominates) and ranging (vol-mean-revert).
  An HMM on a vol-aware feature vector identifies these without supervision.
- 2 states = trending vs. ranging. We label whichever state has higher mean |r|
  as "trending"; the strategy will then only take MC/Markov directional signals
  in that regime.

Cost amortization:
- We fit ONCE on a training prefix (or refit_every N bars), then call .predict()
  for ongoing classification. This avoids the per-bar refits that made Phase 1's
  signal generator slow.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RegimeModel:
    hmm: object                     # fitted hmmlearn.GaussianHMM
    trending_state: int             # which state index is "trending"
    feature_cols: tuple[str, ...]


def _features(close: pd.Series) -> pd.DataFrame:
    log_r = np.log(close).diff()
    return pd.DataFrame(
        {"log_r": log_r, "abs_log_r": log_r.abs()},
        index=close.index,
    ).dropna()


def fit_regime(close: pd.Series, n_states: int = 2, seed: int = 0) -> RegimeModel:
    """Fit an HMM on the supplied close series. Returns a model usable by classify()."""
    from hmmlearn.hmm import GaussianHMM

    feats = _features(close)
    X = feats.values
    hmm = GaussianHMM(
        n_components=n_states, covariance_type="diag", n_iter=200,
        random_state=seed, tol=1e-3,
    )
    hmm.fit(X)

    # Identify which state has the largest mean |log_r| -> "trending"
    mean_abs_per_state = hmm.means_[:, feats.columns.get_loc("abs_log_r")]
    trending_state = int(np.argmax(mean_abs_per_state))
    return RegimeModel(hmm=hmm, trending_state=trending_state,
                       feature_cols=tuple(feats.columns))


def classify(close: pd.Series, model: RegimeModel) -> pd.Series:
    """Per-bar regime label: 1 = trending, 0 = ranging."""
    feats = _features(close)
    states = model.hmm.predict(feats.values)
    is_trending = (states == model.trending_state).astype(np.int8)
    out = pd.Series(0, index=close.index, name="regime", dtype=np.int8)
    out.loc[feats.index] = is_trending
    return out
