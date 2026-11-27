from typing import Annotated, Any

from pandas import Series

from pandas_ta_classic import Imports
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import Bound, Flag, nan_on_short_input


@nan_on_short_input
def rolling_max(
    close: Series,
    length: Annotated[int, Bound(gt=0)] = 30,
    talib: Annotated[bool, Flag()] = False,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Rolling Maximum over *length* periods (TA-Lib: MAX)."""
    close = verify_series(close, length)
    offset = get_offset(offset)
    if close is None:
        return None
    if Imports["talib"] and talib:
        from talib import MAX

        result = MAX(close, length)
    else:
        result = close.rolling(length).max()
    result = apply_offset(result, offset)
    result = apply_fill(result, **kwargs)
    result.name = f"MAX_{length}"
    result.category = "math"
    return result


rolling_max.__doc__ = """Rolling Maximum over *length* periods (TA-Lib: MAX)

Calculation:
    MAX = max(close, length)

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
