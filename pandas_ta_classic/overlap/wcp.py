# -*- coding: utf-8 -*-
# Weighted Close Price (WCP)
from typing import Any, Optional
from pandas import Series
from pandas_ta_classic import Imports
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series


def wcp(
    high: Series,
    low: Series,
    close: Series,
    talib: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Weighted Closing Price (WCP)"""
    # Validate Arguments
    high = verify_series(high)
    low = verify_series(low)
    close = verify_series(close)
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    if high is None or low is None or close is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from talib import WCLPRICE

        wcp = WCLPRICE(high, low, close)
    else:
        wcp = (high + low + 2 * close) / 4

    # Offset
    wcp = apply_offset(wcp, offset)

    wcp = apply_fill(wcp, **kwargs)

    # Name & Category
    wcp.name = "WCP"
    wcp.category = "overlap"

    return wcp


wcp.__doc__ = """Weighted Closing Price (WCP)

Weighted Closing Price is the weighted price given: high, low
and double the close.

Sources:
    https://www.fmlabs.com/reference/default.htm?url=WeightedCloses.htm

Calculation:
    WCP = (2 * close + high + low) / 4

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
