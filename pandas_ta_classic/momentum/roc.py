# -*- coding: utf-8 -*-
# Rate of Change (ROC)
from typing import Any, Optional
from pandas import Series
from .mom import mom
from pandas_ta_classic import Imports
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series


def roc(
    close: Series,
    length: Optional[int] = None,
    scalar: Optional[float] = None,
    talib: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Rate of Change (ROC)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 10
    scalar = float(scalar) if scalar and scalar > 0 else 100
    close = verify_series(close, length)
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    if close is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_tal:
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
    length (int): It's period. Default: 1
    scalar (float): How much to magnify. Default: 100
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
