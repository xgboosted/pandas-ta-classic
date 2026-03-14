# -*- coding: utf-8 -*-
# Fibonacci Weighted Moving Average (FWMA)
from typing import Any, Optional
from pandas import Series
from pandas_ta_classic.utils import fibonacci, get_offset, verify_series
from pandas_ta_classic.utils._core import _sliding_weighted_ma


def fwma(
    close: Series,
    length: Optional[int] = None,
    asc: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Fibonacci's Weighted Moving Average (FWMA)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 10
    asc = asc if asc else True
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result
    fibs = fibonacci(n=length, weighted=True)
    fwma = _sliding_weighted_ma(close, length, fibs)

    # Offset
    if offset != 0:
        fwma = fwma.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        fwma.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        if "fill_method" in kwargs:

            if kwargs["fill_method"] == "ffill":

                fwma.ffill(inplace=True)

            elif kwargs["fill_method"] == "bfill":

                fwma.bfill(inplace=True)

    # Name & Category
    fwma.name = f"FWMA_{length}"
    fwma.category = "overlap"

    return fwma


fwma.__doc__ = """Fibonacci's Weighted Moving Average (FWMA)

Fibonacci's Weighted Moving Average is similar to a Weighted Moving Average
(WMA) where the weights are based on the Fibonacci Sequence.

Source: Kevin Johnson

Calculation:
    Default Inputs:
        length=10,

    def weights(w):
        def _compute(x):
            return np.dot(w * x)
        return _compute

    fibs = utils.fibonacci(length - 1)
    FWMA = close.rolling(length)_.apply(weights(fibs), raw=True)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 10
    asc (bool): Recent values weigh more. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
