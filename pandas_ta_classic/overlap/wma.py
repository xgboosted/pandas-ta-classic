# Weighted Moving Average (WMA)
from typing import Any

import numpy as np
from pandas import Series

from pandas_ta_classic import Imports
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series, weights
from pandas_ta_classic.utils._core import _bool_param, _pos_int, nan_on_short_input


@nan_on_short_input
def wma(
    close: Series,
    length: int | None = None,
    asc: bool | None = None,
    talib: bool | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Weighted Moving Average (WMA)"""
    # Validate Arguments
    length = _pos_int(length, 10, "length")
    asc = _bool_param(asc, True, "asc")
    close = verify_series(close, length)
    offset = get_offset(offset)
    mode_talib = _bool_param(talib, False, "talib")

    if close is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_talib and asc:  # TA-Lib WMA has no descending weights
        from talib import WMA

        wma = WMA(close, length)
    else:
        total_weight = 0.5 * length * (length + 1)
        w = np.arange(1, length + 1)
        if not asc:
            w = w[::-1]

        close_ = close.rolling(length, min_periods=length)
        wma = close_.apply(weights(w), raw=True) / total_weight

    # Offset
    wma = apply_offset(wma, offset)

    wma = apply_fill(wma, **kwargs)

    # Name & Category
    wma.name = f"WMA_{length}"
    wma.category = "overlap"

    return wma


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
    asc (bool): True: recent values weigh more. False: older values weigh
        more (computed natively, even with talib=True). Default: True
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: False
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
