# Rate of Change Ratio * 100 (ROCR100)
from typing import Annotated, Any

from pandas import Series

from pandas_ta_classic import Imports
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import Bound, Flag, _positive_inputs, nan_on_short_input


@nan_on_short_input
@_positive_inputs("close")
def rocr100(
    close: Series,
    length: Annotated[int, Bound(gt=0)] = 10,
    talib: Annotated[bool, Flag()] = False,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Rate of Change Ratio * 100 (ROCR100)"""
    # Validate Arguments
    close = verify_series(close, length)
    offset = get_offset(offset)
    mode_talib = talib

    if close is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_talib:
        from talib import ROCR100 as TAROCR100

        rocr100_ = TAROCR100(close, length)
    else:
        rocr100_ = 100 * (close / close.shift(length))

    # Offset
    rocr100_ = apply_offset(rocr100_, offset)

    rocr100_ = apply_fill(rocr100_, **kwargs)

    # Name and Categorize it
    rocr100_.name = f"ROCR100_{length}"
    rocr100_.category = "momentum"

    return rocr100_


rocr100.__doc__ = """Rate of Change Ratio * 100 (ROCR100)

Rate of Change Ratio * 100 measures the ratio of the current price to the
price n periods ago, scaled by 100. A value of 100 means no change.

ROCR100 = 100 * (close / close[n])

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
    pd.Series: ROCR100 values.
"""
