# -*- coding: utf-8 -*-
# Volume Oscillator (VOSC)
from typing import Any, Optional

from pandas import Series

from pandas_ta_classic.overlap.sma import sma
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series


def vosc(
    volume: Series,
    fast: Optional[int] = None,
    slow: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Volume Oscillator (VOSC)"""
    # Validate Arguments
    fast = int(fast) if fast and fast > 0 else 14
    slow = int(slow) if slow and slow > 0 else 28
    if fast > slow:
        fast, slow = slow, fast
    volume = verify_series(volume, slow)
    offset = get_offset(offset)

    if volume is None:
        return None

    # Calculate Result
    fast_sma = sma(volume, length=fast)
    slow_sma = sma(volume, length=slow)

    if fast_sma is None or slow_sma is None:
        return None

    vosc_ = 100 * (fast_sma - slow_sma) / slow_sma

    # Offset
    vosc_ = apply_offset(vosc_, offset)

    vosc_ = apply_fill(vosc_, **kwargs)

    # Name and Categorize it
    vosc_.name = f"VOSC_{fast}_{slow}"
    vosc_.category = "volume"

    return vosc_


vosc.__doc__ = """Volume Oscillator (VOSC)

The Volume Oscillator measures the difference between two volume moving
averages (fast and slow SMAs) as a percentage of the slow SMA.
Rising VOSC indicates increasing volume momentum.

VOSC = 100 * (SMA(volume, fast) - SMA(volume, slow)) / SMA(volume, slow)

Sources:
    https://school.stockcharts.com/doku.php?id=technical_indicators:volume_oscillator_vo

Args:
    volume (pd.Series): Volume series.
    fast (int): Fast SMA period. Default: 14
    slow (int): Slow SMA period. Default: 28
    offset (int): Result offset. Default: 0

Returns:
    pd.Series: VOSC values.
"""
