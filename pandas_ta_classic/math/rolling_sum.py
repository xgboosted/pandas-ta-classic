from typing import Annotated, Any

from pandas import Series

from pandas_ta_classic import Imports
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import Bound, Flag, nan_on_short_input


@nan_on_short_input
def rolling_sum(
    close: Series,
    length: Annotated[int, Bound(gt=0)] = 30,
    talib: Annotated[bool, Flag()] = False,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Rolling Summation over *length* periods (TA-Lib: SUM)."""
    close = verify_series(close, length)
    offset = get_offset(offset)
    if close is None:
        return None
    if Imports["talib"] and talib:
        from talib import SUM

        result = SUM(close, length)
    else:
        result = close.rolling(length).sum()
    result = apply_offset(result, offset)
    result = apply_fill(result, **kwargs)
    result.name = f"SUM_{length}"
    result.category = "math"
    return result


rolling_sum.__doc__ = """Rolling Summation over *length* periods (TA-Lib: SUM)

Calculation:
    SUM = sum(close, length)

Args:
    close (pd.Series): Series of input values
    length (int): The window length. Default: 30
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: False
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
