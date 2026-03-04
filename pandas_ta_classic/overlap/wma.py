# -*- coding: utf-8 -*-
# Weighted Moving Average (WMA)
from typing import Any, Optional
from pandas import Series
from pandas_ta_classic import Imports
from pandas_ta_classic.utils import (
    _get_tal_mode,
    _finalize,
    _sliding_weighted_ma,
    get_offset,
    verify_series,
)


def wma(
    close: Series,
    length: Optional[int] = None,
    asc: Optional[bool] = None,
    talib: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Weighted Moving Average (WMA)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 10
    asc = asc if asc else True
    close = verify_series(close, length)
    offset = get_offset(offset)
    mode_tal = _get_tal_mode(talib)

    if close is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from talib import WMA

        wma = WMA(close, length)
    else:
        from numpy import arange as npArange

        total_weight = 0.5 * length * (length + 1)
        weights_ = npArange(1, length + 1, dtype=float)
        w = weights_ if asc else weights_[::-1].copy()
        wma = _sliding_weighted_ma(close, length, w / total_weight)

    return _finalize(wma, offset, f"WMA_{length}", "overlap", **kwargs)


wma.__doc__ = """Weighted Moving Average (WMA)

The Weighted Moving Average where the weights are linearly increasing and
the most recent data has the heaviest weight.

Sources:
    https://en.wikipedia.org/wiki/Moving_average#Weighted_moving_average

Calculation:
    Default Inputs:
        length=10, asc=True
    total_weight = 0.5 * length * (length + 1)
    weights_ = [1, 2, ..., length + 1]  # Ascending
    weights = weights if asc else weights[::-1]

    def linear_weights(w):
        def _compute(x):
            return (w * x).sum() / total_weight
        return _compute

    WMA = close.rolling(length)_.apply(linear_weights(weights), raw=True)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 10
    asc (bool): Recent values weigh more. Default: True
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
