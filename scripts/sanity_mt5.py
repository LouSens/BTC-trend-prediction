"""Probe the MT5 terminal connection and print account/symbol info.

Run this once after installing MetaTrader5 and logging into Exness demo:

    python scripts/sanity_mt5.py
"""
from __future__ import annotations

import json
import sys

from mcmc_cuda.data.mt5_loader import sanity_check


def main() -> int:
    try:
        info = sanity_check()
    except Exception as e:
        print(f"MT5 sanity check FAILED: {e}", file=sys.stderr)
        return 1

    print(json.dumps(info, indent=2, default=str))
    if not info.get("is_demo"):
        print(
            "\nWARNING: connected account is NOT a demo account. "
            "The broker bridge will refuse to send orders.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
