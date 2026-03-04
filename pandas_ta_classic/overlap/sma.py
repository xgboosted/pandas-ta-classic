# -*- coding: utf-8 -*-
# Simple Moving Average (SMA)
from typing import Any, Optional
from pandas import Series
from pandas_ta_classic import Imports
from pandas_ta_classic.utils import (
    _get_tal_mode,
    _get_min_periods,
    _finalize,
    get_offset,
    verify_series,
)


def sma(
    close: Series,
    length: Optional[int] = None,
    talib: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Simple Moving Average (SMA)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 10
    min_periods = _get_min_periods(kwargs, length)
    close = verify_series(close, max(length, min_periods))
    offset = get_offset(offset)
    mode_tal = _get_tal_mode(talib)

    if close is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from talib import SMA

        sma = SMA(close, length)
    else:
        sma = close.rolling(length, min_periods=min_periods).mean()

    return _finalize(sma, offset, f"SMA_{length}", "overlap", **kwargs)


sma.__doc__ = """Simple Moving Average (SMA)

The Simple Moving Average is the classic moving average that is the equally
weighted average over n periods.

Sources:
    https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/simple-moving-average-sma/

Calculation:
    Default Inputs:
        length=10
    SMA = SUM(close, length) / length

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 10
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    adjust (bool): Default: True
    presma (bool, optional): If True, uses SMA for initial value.
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
