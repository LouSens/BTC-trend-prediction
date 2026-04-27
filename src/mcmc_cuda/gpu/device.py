"""GPU runtime helpers, with a CPU-graceful fallback when CuPy isn't installed."""
from __future__ import annotations

import os

_CUPY = None
_HAS_CUDA = False


def get_xp():
    """Return CuPy if available and a CUDA device is visible, else NumPy.

    The MCMC_FORCE_CPU=1 env var forces NumPy regardless — useful for unit
    tests in CI where no GPU is present.
    """
    global _CUPY, _HAS_CUDA
    if os.environ.get("MCMC_FORCE_CPU") == "1":
        import numpy as np
        return np

    if _CUPY is not None:
        return _CUPY

    try:
        import cupy as cp

        _ = cp.cuda.runtime.getDeviceCount()
        _CUPY = cp
        _HAS_CUDA = True
        return cp
    except Exception:
        import numpy as np
        return np


def has_cuda() -> bool:
    get_xp()
    return _HAS_CUDA


def free_vram_bytes() -> int | None:
    """Free VRAM in bytes, or None if CUDA isn't available."""
    if not has_cuda():
        return None
    import cupy as cp

    free, _ = cp.cuda.runtime.memGetInfo()
    return int(free)
