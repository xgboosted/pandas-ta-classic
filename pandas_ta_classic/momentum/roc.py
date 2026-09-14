# Rate of Change (ROC)
from typing import Any

from pandas import Series

from pandas_ta_classic import Imports
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _bool_param, _pos_float, _pos_int

from .mom import mom


def roc(
    close: Series,
    length: int | None = None,
    scalar: float | None = None,
    talib: bool | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Rate of Change (ROC)"""
    # Validate Arguments
    length = _pos_int(length, 10, "length")
    scalar = _pos_float(scalar, 100, "scalar")
    close = verify_series(close, length)
    offset = get_offset(offset)
    mode_talib = _bool_param(talib, False, "talib")

    if close is None:
        return None

    # Calculate Result
    # TA-Lib cannot express a non-default scalar; run natively instead of ignoring it
    if Imports["talib"] and mode_talib and scalar == 100:
        from talib import ROC

        roc = ROC(close, length)
    else:
        roc = scalar * mom(close=close, length=length) / close.shift(length)

    # Offset
    roc = apply_offset(roc, offset)

    roc = apply_fill(roc, **kwargs)

    # Name and Categorize it
    roc.name = f"ROC_{length}"
    roc.category = "momentum"

    return roc


roc.__doc__ = """Rate of Change (ROC)

Rate of Change is an indicator is also referred to as Momentum (yeah, confusingly).
It is a pure momentum oscillator that measures the percent change in price with the
previous price 'n' (or length) periods ago.

Sources:
    https://www.tradingview.com/wiki/Rate_of_Change_(ROC)

Calculation:
    Default Inputs:
        length=1
    MOM = Momentum
    ROC = 100 * MOM(close, length) / close.shift(length)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 10
    scalar (float): How much to magnify. Default: 100
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: False
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
