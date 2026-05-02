"""UTC trading-session classification for XAUUSD scalping.

Session definitions (UTC, year-round; we accept the daylight-savings drift
since gold liquidity is much more about clock hour than wall-clock):

    asia     22:00 - 07:00  thin, range-bound, wide spreads
    london   07:00 - 12:00  liquid, directional
    overlap  12:00 - 15:00  London + NY overlap, deepest liquidity
    ny       15:00 - 21:00  NY-only, still active but thinning
    dead     21:00 - 22:00  buffer, avoid scalps

The boundaries are deliberately conservative: we want momentum scalps to
fire during london+overlap and avoid the asia chop and the late-NY drift.

Anything that needs session info should use `label_sessions(index)` once
and then index into that frame, rather than calling _hour_to_session per row.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LIQUID_SESSIONS: tuple[str, ...] = ("london", "overlap", "ny")
PREFERRED_SCALP_SESSIONS: tuple[str, ...] = ("london", "overlap")


@dataclass(frozen=True)
class SessionWindow:
    name: str
    start_hour: int      # inclusive UTC hour
    end_hour: int        # exclusive UTC hour; may wrap past midnight


# Order matters: overlap is checked before london/ny so it wins the priority.
SESSION_WINDOWS: tuple[SessionWindow, ...] = (
    SessionWindow("overlap", 12, 15),
    SessionWindow("london",   7, 12),
    SessionWindow("ny",      15, 21),
    SessionWindow("dead",    21, 22),
    SessionWindow("asia",    22, 31),  # 22..24 + 0..7 handled via wrap below
)


def _hour_to_session(hour: int) -> str:
    h = hour % 24
    for w in SESSION_WINDOWS:
        if w.start_hour <= w.end_hour:
            if w.start_hour <= h < w.end_hour:
                return w.name
        else:
            if h >= w.start_hour or h < (w.end_hour % 24):
                return w.name
    # Asia wraps midnight: catch h in [0..7) here.
    if 0 <= h < 7:
        return "asia"
    return "asia"


def label_sessions(index: pd.DatetimeIndex) -> pd.Series:
    """Vectorized mapping of a UTC DatetimeIndex to a categorical session label."""
    if index.tz is None:
        # Treat naive timestamps as UTC; this matches `data/loader.py`.
        idx = index.tz_localize("UTC")
    else:
        idx = index.tz_convert("UTC")

    hours = idx.hour.to_numpy()
    out = np.empty(hours.shape, dtype=object)
    for i, h in enumerate(hours):
        out[i] = _hour_to_session(int(h))
    return pd.Series(out, index=index, name="session", dtype="object")


def is_liquid(session: str) -> bool:
    return session in LIQUID_SESSIONS


def is_preferred_scalp(session: str) -> bool:
    return session in PREFERRED_SCALP_SESSIONS


def session_mask(index: pd.DatetimeIndex, allowed: tuple[str, ...]) -> pd.Series:
    """Boolean mask True iff the bar's session is in `allowed`."""
    s = label_sessions(index)
    return s.isin(allowed).rename("session_ok")
