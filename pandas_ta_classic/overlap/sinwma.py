# -*- coding: utf-8 -*-
# Sine Weighted Moving Average (SINWMA)
from typing import Any, Optional
from numpy import pi as npPi
from numpy import sin as npSin
from pandas import Series
from pandas_ta_classic.utils import (
    _finalize,
    _sliding_weighted_ma,
    get_offset,
    verify_series,
)


def sinwma(
    close: Series,
    length: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Sine Weighted Moving Average (SINWMA) by Everget of TradingView"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 14
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result
    import numpy as np

    sines = np.array([npSin((i + 1) * npPi / (length + 1)) for i in range(length)])
    w = sines / sines.sum()
    sinwma = _sliding_weighted_ma(close, length, w)

    return _finalize(sinwma, offset, f"SINWMA_{length}", "overlap", **kwargs)


sinwma.__doc__ = """Sine Weighted Moving Average (SWMA)

A weighted average using sine cycles. The middle term(s) of the average have the
highest weight(s).

Source:
    https://www.tradingview.com/script/6MWFvnPO-Sine-Weighted-Moving-Average/
    Author: Everget (https://www.tradingview.com/u/everget/)

Calculation:
    Default Inputs:
        length=10

    def weights(w):
        def _compute(x):
            return np.dot(w * x)
        return _compute

    sines = Series([sin((i + 1) * pi / (length + 1)) for i in range(0, length)])
    w = sines / sines.sum()
    SINWMA = close.rolling(length, min_periods=length).apply(weights(w), raw=True)

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
