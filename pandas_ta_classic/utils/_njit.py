"""Numba njit decorator with graceful fallback.

When numba is installed (``pip install numba``), re-exports the real
``njit``.  Otherwise provides a no-op decorator so that the same
``@njit(cache=True)`` syntax works everywhere without ImportError.
"""

import inspect
import os

try:
    from numba import njit as _numba_njit
except ImportError:

    def njit(*args, **kwargs):  # type: ignore[misc]
        """No-op decorator mimicking ``numba.njit``."""

        def _wrap(f):
            return f

        return _wrap if not args or not callable(args[0]) else args[0]

else:

    def njit(*args, **kwargs):  # type: ignore[misc]
        """``numba.njit``, without the on-disk cache for code imported from a zip archive.

        numba places the cache of a function whose source is not a file in the
        user cache directory, ignoring ``NUMBA_CACHE_DIR``, and raises
        ``PermissionError`` on the first call when that directory is read-only.
        The cache only saves compile time, so such functions compile in memory.
        """

        def decorate(f):
            options = dict(kwargs)
            if options.get("cache") and not os.path.isfile(inspect.getsourcefile(f) or ""):
                options["cache"] = False
            return _numba_njit(**options)(f)

        return decorate(args[0]) if args and callable(args[0]) else decorate


__all__ = ["njit"]
