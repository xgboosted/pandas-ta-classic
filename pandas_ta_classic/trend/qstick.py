# Q Stick (QSTICK)
from typing import Any

from pandas import Series

from pandas_ta_classic.overlap.ma import ma
from pandas_ta_classic.utils import (
    apply_fill,
    apply_offset,
    get_offset,
    verify_series,
)
from pandas_ta_classic.utils._core import _pos_int, _str_param, nan_on_short_input


@nan_on_short_input
def qstick(
    open_: Series,
    close: Series,
    length: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Q Stick"""
    # Validate Arguments
    length = _pos_int(length, 10, "length")
    mamode = _str_param(kwargs.pop("ma", None), "sma", "ma", choices=ma())
    # The fill applies to the result only; the other options go to the chosen MA (ema's adjust, talib)
    fill_kwargs = {key: kwargs.pop(key) for key in ("fillna", "fill_method") if key in kwargs}
    open_ = verify_series(open_, length)
    close = verify_series(close, length)
    offset = get_offset(offset)

    if open_ is None or close is None:
        return None

    # Calculate Result
    # A doji's body is 0, and the average of the last `length` bodies is what
    # this reports -- nothing divides by it, so non_zero_range()'s epsilon only
    # ever reached the output as 2.2e-16 of body where there was none.
    diff = close - open_

    qstick = ma(mamode, diff, length=length, **kwargs)

    if qstick is None:
        return None

    # Offset
    qstick = apply_offset(qstick, offset)

    qstick = apply_fill(qstick, **fill_kwargs)

    # Name and Categorize it
    qstick.name = f"QS_{length}"
    qstick.category = "trend"

    return qstick


qstick.__doc__ = """Q Stick

The Q Stick indicator, developed by Tushar Chande, attempts to quantify and
identify trends in candlestick charts.

Sources:
    https://library.tradingtechnologies.com/trade/chrt-ti-qstick.html

Calculation:
    Default Inputs:
        length=10
    xMA is any MA of help(ta.ma); sma by default
    qstick = xMA(close - open, length)

Args:
    open_ (pd.Series): Series of 'open's
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 10
    ma (str): The moving average to use, see help(ta.ma). Default: 'sma'
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
