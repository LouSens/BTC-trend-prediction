"""Project paths and runtime configuration."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"
RUNS_DIR = ROOT / "runs"

for _d in (RAW_DIR, PROCESSED_DIR, ARTIFACTS_DIR, RUNS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT / ".env", override=False)

SYMBOL = os.environ.get("MCMC_SYMBOL", "XAUUSD")
DEFAULT_TIMEFRAME = os.environ.get("MCMC_TIMEFRAME", "M15")
