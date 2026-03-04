# -*- coding: utf-8 -*-
# Long Run (LONG_RUN)
from typing import Any, Optional
from pandas import Series
from .decreasing import decreasing
from .increasing import increasing
from pandas_ta_classic.utils import _finalize, get_offset, verify_series


def long_run(
    fast: Series,
    slow: Series,
    length: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Long Run"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 2
    fast = verify_series(fast, length)
    slow = verify_series(slow, length)
    offset = get_offset(offset)

    if fast is None or slow is None:
        return None

    # Calculate Result
    pb = increasing(fast, length) & decreasing(
        slow, length
    )  # potential bottom or bottom
    bi = increasing(fast, length) & increasing(
        slow, length
    )  # fast and slow are increasing
    long_run = pb | bi

    return _finalize(long_run, offset, f"LR_{length}", "trend", **kwargs)


long_run.__doc__ = """Long Run

Identifies potential long (bullish) trend conditions by detecting when the fast 
moving average is increasing while the slow moving average is either decreasing 
(potential bottom) or also increasing (confirmed uptrend).

Sources:
    Used in AMAT (Archer Moving Averages Trends) indicator

Calculation:
    Default Inputs:
        length=2
    
    PB = INCREASING(fast, length) AND DECREASING(slow, length)  # Potential bottom
    BI = INCREASING(fast, length) AND INCREASING(slow, length)  # Both increasing
    LONG_RUN = PB OR BI

Args:
    fast (pd.Series): Fast moving average series
    slow (pd.Series): Slow moving average series
    length (int): Lookback period. Default: 2
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated (boolean).
"""
