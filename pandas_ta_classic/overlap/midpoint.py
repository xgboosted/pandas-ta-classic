# -*- coding: utf-8 -*-
# Midpoint (MIDPOINT)
from typing import Any, Optional
from pandas import Series
from pandas_ta_classic import Imports
from pandas_ta_classic.utils import apply_offset, get_offset, verify_series


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
    min_periods = (
        int(kwargs["min_periods"])
        if "min_periods" in kwargs and kwargs["min_periods"] is not None
        else length
    )
    close = verify_series(close, max(length, min_periods))
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

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

    # Offset
    midpoint = apply_offset(midpoint, offset, **kwargs)

    # Name and Categorize it
    midpoint.name = f"MIDPOINT_{length}"
    midpoint.category = "overlap"

    return midpoint


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
    length (int): Its period. Default: 2
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
