# -*- coding: utf-8 -*-
# Midprice (MIDPRICE)
from typing import Any, Optional
from pandas import Series
from pandas_ta_classic import Imports
from pandas_ta_classic.utils import (
    _get_min_periods,
    _finalize,
    get_offset,
    verify_series,
)


def midprice(
    high: Series,
    low: Series,
    length: Optional[int] = None,
    talib: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Midprice"""
    # Validate arguments
    length = int(length) if length and length > 0 else 2
    min_periods = _get_min_periods(kwargs, length)
    _length = max(length, min_periods)
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    if high is None or low is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from talib import MIDPRICE

        midprice = MIDPRICE(high, low, length)
    else:
        lowest_low = low.rolling(length, min_periods=min_periods).min()
        highest_high = high.rolling(length, min_periods=min_periods).max()
        midprice = 0.5 * (lowest_low + highest_high)

    return _finalize(midprice, offset, f"MIDPRICE_{length}", "overlap", **kwargs)


midprice.__doc__ = """Midpoint Price Over Period (MIDPRICE)

MIDPRICE calculates the midpoint between the highest high and lowest low 
over a specified period. Similar to MIDPOINT but uses high and low prices 
instead of close prices. This provides a measure of the center of the 
price range and is useful for identifying equilibrium levels.

Sources:
    https://www.tradingview.com/support/solutions/43000594684-midprice/
    https://ta-lib.org/function.html?name=MIDPRICE

Calculation:
    Default Inputs:
        length=2
    
    LOWEST_LOW = MIN(low, length)
    HIGHEST_HIGH = MAX(high, length)
    MIDPRICE = (LOWEST_LOW + HIGHEST_HIGH) / 2

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
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
