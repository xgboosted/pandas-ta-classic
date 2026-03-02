# -*- coding: utf-8 -*-
# Hull Moving Average (HMA)
from typing import Any, Optional
from numpy import sqrt as npSqrt
from pandas import Series
from .wma import wma
from pandas_ta_classic.utils import _finalize, get_offset, verify_series


def hma(
    close: Series,
    length: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Hull Moving Average (HMA)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 10
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result
    half_length = length // 2
    sqrt_length = int(npSqrt(length))

    wmaf = wma(close=close, length=half_length)
    wmas = wma(close=close, length=length)
    hma = wma(close=2 * wmaf - wmas, length=sqrt_length)

    return _finalize(hma, offset, f"HMA_{length}", "overlap", **kwargs)


hma.__doc__ = """Hull Moving Average (HMA)

The Hull Exponential Moving Average attempts to reduce or remove lag in moving
averages.

Sources:
    https://alanhull.com/hull-moving-average

Calculation:
    Default Inputs:
        length=10
    WMA = Weighted Moving Average
    half_length = int(0.5 * length)
    sqrt_length = int(sqrt(length))

    wmaf = WMA(close, half_length)
    wmas = WMA(close, length)
    HMA = WMA(2 * wmaf - wmas, sqrt_length)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 10
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
