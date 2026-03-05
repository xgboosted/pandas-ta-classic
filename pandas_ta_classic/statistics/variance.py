# -*- coding: utf-8 -*-
# Variance (VARIANCE)
from typing import Any, Optional

from pandas import Series

from pandas_ta_classic import Imports
from pandas_ta_classic.utils import (
    _get_tal_mode,
    _get_min_periods,
    _finalize,
    get_offset,
    verify_series,
)


def variance(
    close: Series,
    length: Optional[int] = None,
    ddof: Optional[int] = None,
    talib: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Variance"""
    # Validate Arguments
    length = int(length) if length and length > 1 else 30
    ddof = int(ddof) if isinstance(ddof, int) and ddof >= 0 and ddof < length else 0
    min_periods = _get_min_periods(kwargs, length)
    close = verify_series(close, max(length, min_periods))
    offset = get_offset(offset)
    mode_tal = _get_tal_mode(talib)

    if close is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from talib import VAR

        variance = VAR(close, length)
    else:
        variance = close.rolling(length, min_periods=min_periods).var(ddof=ddof)

    return _finalize(variance, offset, f"VAR_{length}", "statistics", **kwargs)


variance.__doc__ = """Rolling Variance

Sources:

Calculation:
    Default Inputs:
        length=30
    VARIANCE = close.rolling(length).var()

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 30 (TA-Lib default: 5)
    ddof (int): Delta Degrees of Freedom.
                The divisor used in calculations is N - ddof,
                where N represents the number of elements. Default: 0
                (population variance, matches TA-Lib)
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
