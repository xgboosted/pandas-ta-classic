# -*- coding: utf-8 -*-
# Double Exponential Moving Average (DEMA)
from typing import Any, Optional
from pandas import Series
from .ema import ema
from pandas_ta_classic import Imports
from pandas_ta_classic.utils import _finalize, get_offset, verify_series


def dema(
    close: Series,
    length: Optional[int] = None,
    talib: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Double Exponential Moving Average (DEMA)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 10
    close = verify_series(close, length)
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    if close is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from talib import DEMA

        dema = DEMA(close, length)
    else:
        ema1 = ema(close=close, length=length)
        ema2 = ema(close=ema1, length=length)
        dema = 2 * ema1 - ema2

    return _finalize(dema, offset, f"DEMA_{length}", "overlap", **kwargs)


dema.__doc__ = """Double Exponential Moving Average (DEMA)

The Double Exponential Moving Average attempts to a smoother average with less
lag than the normal Exponential Moving Average (EMA).

Sources:
    https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/double-exponential-moving-average-dema/

Calculation:
    Default Inputs:
        length=10
    EMA = Exponential Moving Average
    ema1 = EMA(close, length)
    ema2 = EMA(ema1, length)

    DEMA = 2 * ema1 - ema2

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 10
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
