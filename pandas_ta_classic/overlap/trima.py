# Triangular Moving Average (TRIMA)
from typing import Any

from pandas import Series

from pandas_ta_classic import Imports
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _bool_param, _pos_int, nan_on_short_input

from .sma import sma


@nan_on_short_input
def trima(
    close: Series,
    length: int | None = None,
    talib: bool | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Triangular Moving Average (TRIMA)"""
    # Validate Arguments
    length = _pos_int(length, 10, "length")
    close = verify_series(close, length)
    offset = get_offset(offset)
    mode_talib = _bool_param(talib, False, "talib")

    if close is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_talib:
        from talib import TRIMA

        trima = TRIMA(close, length)
    else:
        # TA-Lib TRIMA: SMA(SMA(close, n/2 + 1), n/2) for even n, and
        # SMA(SMA(close, (n+1)/2), (n+1)/2) for odd n.  The second window is
        # (length + 1) // 2; using length // 2 + 1 for it built a triangle one
        # bar too wide for even lengths (e.g. 6x6 instead of 6x5 for n=10).
        len1 = length // 2 + 1
        len2 = (length + 1) // 2
        sma1 = sma(close, length=len1, talib=False)
        if sma1 is None:
            return None
        trima = sma(sma1, length=len2, talib=False)
        if trima is None:
            return None

    # Offset
    trima = apply_offset(trima, offset)

    trima = apply_fill(trima, **kwargs)

    # Name & Category
    trima.name = f"TRIMA_{length}"
    trima.category = "overlap"

    return trima


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
        version. Default: False
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    adjust (bool): Default: True
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
