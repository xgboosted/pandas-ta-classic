# Standard Deviation (STDEV)
from typing import Any

import numpy as np
from pandas import Series

from pandas_ta_classic import Imports
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _bool_param, _pos_int, nan_on_short_input

from .variance import variance


@nan_on_short_input
def stdev(
    close: Series,
    length: int | None = None,
    ddof: int | None = None,
    talib: bool | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Standard Deviation"""
    # Validate Arguments
    length = _pos_int(length, 30, "length", gt=1)  # variance needs at least two rows
    ddof = _pos_int(ddof, 0, "ddof", gt=None, ge=0, lt=length)
    # Same bound as variance(): the error must name stdev, the function called.
    min_periods = _pos_int(kwargs.pop("min_periods", None), length, "min_periods", gt=None, ge=0)
    close = verify_series(close, max(length, min_periods))
    offset = get_offset(offset)
    mode_talib = _bool_param(talib, False, "talib")

    if close is None:
        return None

    # Calculate Result
    # TA-Lib cannot express a non-default ddof; run natively instead of ignoring it
    if Imports["talib"] and mode_talib and ddof == 0:
        from talib import STDDEV

        stdev = STDDEV(close, length)
    else:
        _variance = variance(close=close, length=length, ddof=ddof, min_periods=min_periods, talib=False)
        if _variance is None:
            return None
        stdev = _variance.apply(np.sqrt)

    # Offset
    stdev = apply_offset(stdev, offset)

    stdev = apply_fill(stdev, **kwargs)

    # Name & Category
    stdev.name = f"STDEV_{length}"
    stdev.category = "statistics"

    return stdev


stdev.__doc__ = """Rolling Standard Deviation

Sources:

Calculation:
    Default Inputs:
        length=30
    VAR = Variance
    STDEV = variance(close, length).apply(np.sqrt)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 30
    ddof (int): Delta Degrees of Freedom.
                The divisor used in calculations is N - ddof,
                where N represents the number of elements. Default: 0
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: False
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
