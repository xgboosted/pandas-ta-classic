# -*- coding: utf-8 -*-
# Rate of Change Percentage (ROCP)
from typing import Any, Optional

from pandas import Series

from pandas_ta_classic import Imports
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series


def rocp(
    close: Series,
    length: Optional[int] = None,
    talib: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Rate of Change Percentage (ROCP)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 10
    close = verify_series(close, length)
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    if close is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from talib import ROCP as TAROCP

        rocp_ = TAROCP(close, length)
    else:
        rocp_ = (close - close.shift(length)) / close.shift(length)

    # Offset
    rocp_ = apply_offset(rocp_, offset)

    rocp_ = apply_fill(rocp_, **kwargs)

    # Name and Categorize it
    rocp_.name = f"ROCP_{length}"
    rocp_.category = "momentum"

    return rocp_


rocp.__doc__ = """Rate of Change Percentage (ROCP)

Rate of Change Percentage measures the percentage change in price over a
given period, expressed as a ratio (not multiplied by 100).

ROCP = (close - close[n]) / close[n]

Sources:
    https://www.investopedia.com/terms/r/rateofchange.asp

Args:
    close (pd.Series): Close price series.
    length (int): The period. Default: 10
    talib (bool): Use TA-Lib if installed. Default: True
    offset (int): Result offset. Default: 0

Returns:
    pd.Series: ROCP values.
"""
