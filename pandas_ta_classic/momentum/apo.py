# -*- coding: utf-8 -*-
# Absolute Price Oscillator (APO)
from typing import Any, Optional
from pandas import Series
from pandas_ta_classic import Imports
from pandas_ta_classic.overlap.ma import ma
from pandas_ta_classic.utils import (
    apply_fill,
    apply_offset,
    get_offset,
    tal_ma,
    verify_series,
)


def apo(
    close: Series,
    fast: Optional[int] = None,
    slow: Optional[int] = None,
    mamode: Optional[str] = None,
    talib: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Absolute Price Oscillator (APO)"""
    # Validate Arguments
    fast = int(fast) if fast and fast > 0 else 12
    slow = int(slow) if slow and slow > 0 else 26
    if slow < fast:
        fast, slow = slow, fast
    close = verify_series(close, max(fast, slow))
    mamode = mamode if isinstance(mamode, str) else "sma"
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    if close is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from talib import APO

        apo = APO(close, fast, slow, tal_ma(mamode))
    else:
        fastma = ma(mamode, close, length=fast)
        if fastma is None:
            return None
        slowma = ma(mamode, close, length=slow)
        if slowma is None:
            return None
        apo = fastma - slowma

    # Offset
    apo = apply_offset(apo, offset)

    apo = apply_fill(apo, **kwargs)

    # Name and Categorize it
    apo.name = f"APO_{fast}_{slow}"
    apo.category = "momentum"

    return apo


apo.__doc__ = """Absolute Price Oscillator (APO)

The Absolute Price Oscillator is an indicator used to measure a security's
momentum.  It is simply the difference of two Exponential Moving Averages
(EMA) of two different periods. Note: APO and MACD lines are equivalent.

Sources:
    https://www.tradingtechnologies.com/xtrader-help/x-study/technical-indicator-definitions/absolute-price-oscillator-apo/

Calculation:
    Default Inputs:
        fast=12, slow=26
    SMA = Simple Moving Average
    APO = SMA(close, fast) - SMA(close, slow)

Args:
    close (pd.Series): Series of 'close's
    fast (int): The short period. Default: 12
    slow (int): The long period. Default: 26
    mamode (str): See ```help(ta.ma)```. Default: 'sma'
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
