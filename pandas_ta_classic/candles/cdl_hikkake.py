from typing import Any, Optional

from pandas import Series

from pandas_ta_classic.candles._cdl_math import (
    CandleArrays,
    run_pattern,
)
import numpy as np


def _hikkake_is_setup(H, L, i):
    """Check if bars at index i form a Hikkake setup (inside bar + breakout direction)."""
    return H[i - 1] < H[i - 2] and L[i - 1] > L[i - 2] and ((H[i] < H[i - 1] and L[i] < L[i - 1]) or (H[i] > H[i - 1] and L[i] > L[i - 1]))


def _hikkake_is_confirmed(pattern_result, pattern_idx, C, H, L, i):
    """Check if bar i confirms a previously detected Hikkake pattern."""
    return i <= pattern_idx + 3 and ((pattern_result > 0 and C[i] > H[pattern_idx - 1]) or (pattern_result < 0 and C[i] < L[pattern_idx - 1]))


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    # Lookback = 5  (no candle settings used)
    lookback = 5
    start_idx = lookback
    if start_idx >= len(out):
        return

    H = ca.high
    L = ca.low
    C = ca.close

    pattern_idx = 0
    pattern_result = 0

    # Warm-up: scan the 3 bars before start_idx to initialize state
    # (i runs from start_idx-3 to start_idx-1)
    for i in range(start_idx - 3, start_idx):
        if _hikkake_is_setup(H, L, i):
            pattern_result = 100 * (1 if H[i] < H[i - 1] else -1)
            pattern_idx = i
        else:
            # Search for confirmation
            if _hikkake_is_confirmed(pattern_result, pattern_idx, C, H, L, i):
                pattern_idx = 0

    # Main loop
    for i in range(start_idx, len(out)):
        if _hikkake_is_setup(H, L, i):
            pattern_result = 100 * (1 if H[i] < H[i - 1] else -1)
            pattern_idx = i
            out[i] = pattern_result
        elif _hikkake_is_confirmed(pattern_result, pattern_idx, C, H, L, i):
            out[i] = pattern_result + 100 * (1 if pattern_result > 0 else -1)
            pattern_idx = 0


def cdl_hikkake(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Hikkake

    A stateful pattern that detects an inside bar (bar 2 has lower high
    and higher low than bar 1) followed by a breakout bar (bar 3) that
    exceeds the inside bar's range. Confirmation occurs within the next
    3 bars when price closes beyond the inside bar's high/low.

    The pattern bar outputs +/-100 and the confirmation bar outputs
    +/-200.

    Args:
        open_: Series of 'open' prices.
        high: Series of 'high' prices.
        low: Series of 'low' prices.
        close: Series of 'close' prices.
        scalar: Multiplier for output values. Default: 100.
        offset: Number of periods to shift the result.

    Returns:
        A Series with pattern signals, or None.

    Example:
        >>> result = cdl_hikkake(df.open, df.high, df.low, df.close)
    """
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_HIKKAKE",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
