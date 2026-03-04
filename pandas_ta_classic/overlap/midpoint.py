# -*- coding: utf-8 -*-
# Midpoint (MIDPOINT)
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


def midpoint(
    close: Series,
    length: Optional[int] = None,
    talib: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Midpoint"""
    # Validate arguments
    length = int(length) if length and length > 0 else 2
    min_periods = _get_min_periods(kwargs, length)
    close = verify_series(close, max(length, min_periods))
    offset = get_offset(offset)
    mode_tal = _get_tal_mode(talib)

    if close is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from talib import MIDPOINT

        midpoint = MIDPOINT(close, length)
    else:
        lowest = close.rolling(length, min_periods=min_periods).min()
        highest = close.rolling(length, min_periods=min_periods).max()
        midpoint = 0.5 * (lowest + highest)

    return _finalize(midpoint, offset, f"MIDPOINT_{length}", "overlap", **kwargs)


midpoint.__doc__ = """Midpoint Over Period (MIDPOINT)

MIDPOINT calculates the midpoint between the highest and lowest values of 
the close price over a specified period. This indicator helps identify the 
center of the price range and can be used to detect potential support and 
resistance levels.

Sources:
    https://www.tradingview.com/support/solutions/43000594683-midpoint/
    https://ta-lib.org/function.html?name=MIDPOINT

Calculation:
    Default Inputs:
        length=2
    
    LOWEST = MIN(close, length)
    HIGHEST = MAX(close, length)
    MIDPOINT = (LOWEST + HIGHEST) / 2

Args:
    close (pd.Series): Series of 'close's
    length (int): Its period. Default: 2 (TA-Lib default: 14)
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    min_periods (int, optional): Minimum number of observations required. Default: length
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
