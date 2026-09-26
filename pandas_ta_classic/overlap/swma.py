# Symmetric Weighted Moving Average (SWMA)
from typing import Any

from pandas import Series

from pandas_ta_classic.utils import (
    apply_fill,
    apply_offset,
    get_offset,
    symmetric_triangle,
    verify_series,
    weights,
)
from pandas_ta_classic.utils._core import _pos_int, nan_on_short_input


@nan_on_short_input
def swma(
    close: Series,
    length: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Symmetric Weighted Moving Average (SWMA)"""
    # Validate Arguments
    length = _pos_int(length, 10, "length")
    close = verify_series(close, length)
    offset = get_offset(offset)
    # A strategy-wide asc (df.ta.strategy(..., asc=False)) is meant for wma/fwma; these
    # weights are symmetric, so it has no meaning here. Drop it rather than forward it to apply_fill.
    kwargs.pop("asc", None)

    if close is None:
        return None

    # Calculate Result
    triangle = symmetric_triangle(length, weighted=True)
    swma = close.rolling(length, min_periods=length).apply(weights(triangle), raw=True)

    # Offset
    swma = apply_offset(swma, offset)

    swma = apply_fill(swma, **kwargs)

    # Name & Category
    swma.name = f"SWMA_{length}"
    swma.category = "overlap"

    return swma


swma.__doc__ = """Symmetric Weighted Moving Average (SWMA)

Symmetric Weighted Moving Average where weights are based on a symmetric
triangle.  For example: n=3 -> [1, 2, 1], n=4 -> [1, 2, 2, 1], etc...
This moving average has variable length in contrast to TradingView's fixed
length of 4.

Source:
    https://www.tradingview.com/study-script-reference/#fun_swma

Calculation:
    Default Inputs:
        length=10

    def weights(w):
        def _compute(x):
            return np.dot(w * x)
        return _compute

    triangle = utils.symmetric_triangle(length - 1)
    SWMA = close.rolling(length)_.apply(weights(triangle), raw=True)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 10
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.

Note: ``asc`` was removed in 0.9.0. The weights are symmetric, so it never
changed the result; a strategy-wide ``asc`` is ignored.
"""
