# Commodity Channel Index (CCI)
from typing import Any

from pandas import Series

from pandas_ta_classic import Imports
from pandas_ta_classic.overlap.hlc3 import hlc3
from pandas_ta_classic.overlap.sma import sma
from pandas_ta_classic.statistics.mad import mad
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _bool_param, _pos_float, _pos_int, nan_on_short_input


@nan_on_short_input
def cci(
    high: Series,
    low: Series,
    close: Series,
    length: int | None = None,
    c: float | None = None,
    talib: bool | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Commodity Channel Index (CCI)"""
    # Validate Arguments
    length = _pos_int(length, 14, "length")
    c = _pos_float(c, 0.015, "c")
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    offset = get_offset(offset)
    mode_talib = _bool_param(talib, False, "talib")

    if high is None or low is None or close is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_talib:
        from talib import CCI

        cci = CCI(high, low, close, length)
    else:
        typical_price = hlc3(high=high, low=low, close=close)
        mean_typical_price = sma(typical_price, length=length, talib=False)
        mad_typical_price = mad(typical_price, length=length)
        if mean_typical_price is None or mad_typical_price is None:
            return None

        cci = typical_price - mean_typical_price
        cci /= c * mad_typical_price

    # Offset
    cci = apply_offset(cci, offset)

    cci = apply_fill(cci, **kwargs)

    # Name and Categorize it
    cci.name = f"CCI_{length}_{c}"
    cci.category = "momentum"

    return cci


cci.__doc__ = """Commodity Channel Index (CCI)

Commodity Channel Index is a momentum oscillator used to primarily identify
overbought and oversold levels relative to a mean.

Sources:
    https://www.tradingview.com/wiki/Commodity_Channel_Index_(CCI)

Calculation:
    Default Inputs:
        length=14, c=0.015
    SMA = Simple Moving Average
    MAD = Mean Absolute Deviation
    tp = typical_price = hlc3 = (high + low + close) / 3
    mean_tp = SMA(tp, length)
    mad_tp = MAD(tp, length)
    CCI = (tp - mean_tp) / (c * mad_tp)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 14
    c (float): Scaling Constant. Default: 0.015
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: False
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
