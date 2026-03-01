# -*- coding: utf-8 -*-
# Time Series Forecast (TSF)
from typing import Any, Optional

from pandas import Series

from pandas_ta_classic import Imports
from pandas_ta_classic.overlap.linreg import linreg
from pandas_ta_classic.utils import (
    _get_tal_mode,
    _finalize,
    get_offset,
    verify_series,
)


def tsf(
    close: Series,
    length: Optional[int] = None,
    talib: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Time Series Forecast"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 14
    close = verify_series(close, length)
    offset = get_offset(offset)
    mode_tal = _get_tal_mode(talib)

    if close is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from talib import TSF as taTSF

        result = Series(taTSF(close, timeperiod=length), index=close.index)
    else:
        result = linreg(close, length=length, tsf=True)

    return _finalize(result, offset, f"TSF_{length}", "overlap", **kwargs)


tsf.__doc__ = """Time Series Forecast (TSF)

The Time Series Forecast projects prices using linear regression.  It equals
the linear regression value at the last bar of each window, i.e. the predicted
next value.  Equivalent to ``linreg(close, length, tsf=True)``.

Sources:
    TA Lib

Calculation:
    Default Inputs:
        length=14
    TSF = slope * length + intercept  (linear regression forecast)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 14
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
