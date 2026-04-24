# -*- coding: utf-8 -*-
"""Numba njit decorator with graceful fallback.

When numba is installed (``pip install numba``), re-exports the real
``njit``.  Otherwise provides a no-op decorator so that the same
``@njit(cache=True)`` syntax works everywhere without ImportError.
"""

try:
    from numba import njit
except ImportError:

    def njit(*args, **kwargs):  # type: ignore[misc]
        """No-op decorator mimicking ``numba.njit``."""

        def _wrap(f):
            return f

        return _wrap if not args or not callable(args[0]) else args[0]


__all__ = ["njit"]
