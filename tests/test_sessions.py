"""Session classification tests."""
from __future__ import annotations

import pandas as pd

from mcmc_cuda.strategy.sessions import (
    LIQUID_SESSIONS,
    PREFERRED_SCALP_SESSIONS,
    is_liquid,
    is_preferred_scalp,
    label_sessions,
    session_mask,
)


def test_label_sessions_assigns_correct_buckets():
    # Hand-picked UTC times spanning every session.
    idx = pd.DatetimeIndex([
        "2024-06-03 00:00",  # asia
        "2024-06-03 06:00",  # asia
        "2024-06-03 08:00",  # london
        "2024-06-03 13:30",  # overlap
        "2024-06-03 16:00",  # ny
        "2024-06-03 21:30",  # dead
        "2024-06-03 22:30",  # asia
    ], tz="UTC")
    out = label_sessions(idx).tolist()
    assert out == ["asia", "asia", "london", "overlap", "ny", "dead", "asia"]


def test_session_mask_filters_to_allowed():
    idx = pd.date_range("2024-06-03 00:00", "2024-06-03 23:00", freq="h", tz="UTC")
    mask = session_mask(idx, PREFERRED_SCALP_SESSIONS)
    # London (07-12) + Overlap (12-15) = 8 hours.
    assert int(mask.sum()) == 8


def test_is_liquid_helpers():
    assert is_liquid("london") and is_liquid("overlap") and is_liquid("ny")
    assert not is_liquid("asia") and not is_liquid("dead")
    assert is_preferred_scalp("london") and is_preferred_scalp("overlap")
    assert not is_preferred_scalp("ny")
    # Sanity: every preferred is also liquid.
    assert set(PREFERRED_SCALP_SESSIONS).issubset(set(LIQUID_SESSIONS))


def test_naive_index_treated_as_utc():
    idx_naive = pd.DatetimeIndex(["2024-06-03 13:00"])
    idx_utc = pd.DatetimeIndex(["2024-06-03 13:00"], tz="UTC")
    assert label_sessions(idx_naive).iloc[0] == label_sessions(idx_utc).iloc[0] == "overlap"
