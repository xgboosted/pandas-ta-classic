# Rate of Change Percentage (ROCP)
from typing import Annotated, Any

from pandas import Series

from pandas_ta_classic import Imports
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import Bound, Flag, _positive_inputs, nan_on_short_input


@nan_on_short_input
@_positive_inputs("close")
def rocp(
    close: Series,
    length: Annotated[int, Bound(gt=0)] = 10,
    talib: Annotated[bool, Flag()] = False,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Rate of Change Percentage (ROCP)"""
    # Validate Arguments
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result
    if Imports["talib"] and talib:
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
    talib (bool): Use TA-Lib if installed. Default: False
    offset (int): Result offset. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: ROCP values.
"""
