# -*- coding: utf-8 -*-
# Pascal Weighted Moving Average (PWMA)
from typing import Any, Optional
from pandas import Series
from pandas_ta_classic.utils import (
    _finalize,
    _sliding_weighted_ma,
    apply_offset,
    get_offset,
    pascals_triangle,
    verify_series,
)


def pwma(
    close: Series,
    length: Optional[int] = None,
    asc: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Pascals Weighted Moving Average (PWMA)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 10
    asc = asc if asc else True
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result
    triangle = pascals_triangle(n=length - 1, weighted=True)
    pwma = _sliding_weighted_ma(close, length, triangle)

    return _finalize(pwma, offset, f"PWMA_{length}", "overlap", **kwargs)


pwma.__doc__ = """Pascal's Weighted Moving Average (PWMA)

Pascal's Weighted Moving Average is similar to a symmetric triangular window
except PWMA's weights are based on Pascal's Triangle.

Source: Kevin Johnson

Calculation:
    Default Inputs:
        length=10

    def weights(w):
        def _compute(x):
            return np.dot(w * x)
        return _compute

    triangle = utils.pascals_triangle(length + 1)
    PWMA = close.rolling(length)_.apply(weights(triangle), raw=True)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period.  Default: 10
    asc (bool): Recent values weigh more. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
