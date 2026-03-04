# -*- coding: utf-8 -*-
# Triangular Moving Average (TRIMA)
from math import ceil, floor
from typing import Any, Optional
from pandas import Series
from .sma import sma
from pandas_ta_classic import Imports
from pandas_ta_classic.utils import _get_tal_mode, _finalize, get_offset, verify_series


def trima(
    close: Series,
    length: Optional[int] = None,
    talib: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Triangular Moving Average (TRIMA)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 10
    close = verify_series(close, length)
    offset = get_offset(offset)
    mode_tal = _get_tal_mode(talib)

    if close is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from talib import TRIMA

        trima = TRIMA(close, length)
    else:
        first_window = ceil(length / 2)
        second_window = floor(length / 2) + 1
        sma1 = sma(close, length=first_window)
        trima = sma(sma1, length=second_window)

    return _finalize(trima, offset, f"TRIMA_{length}", "overlap", **kwargs)


trima.__doc__ = """Triangular Moving Average (TRIMA)

A weighted moving average where the shape of the weights are triangular and the
greatest weight is in the middle of the period.

Sources:
    https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/triangular-moving-average-trima/
    tma = sma(sma(src, ceil(length / 2)), floor(length / 2) + 1)  # Tradingview
    trima = sma(sma(x, n), n)  # Tradingview

Calculation:
    Default Inputs:
        length=10
    SMA = Simple Moving Average
    first_window = ceil(length / 2)
    second_window = floor(length / 2) + 1
    SMA1 = SMA(close, first_window)
    TRIMA = SMA(SMA1, second_window)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 10
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    adjust (bool): Default: True
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
