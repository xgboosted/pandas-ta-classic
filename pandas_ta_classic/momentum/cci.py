# -*- coding: utf-8 -*-
# Commodity Channel Index (CCI)
from typing import Any, Optional
from pandas import Series
from pandas_ta_classic import Imports
from pandas_ta_classic.overlap.hlc3 import hlc3
from pandas_ta_classic.overlap.sma import sma
from pandas_ta_classic.statistics import mad, stdev
from pandas_ta_classic.utils import _finalize, get_offset, verify_series


def cci(
    high: Series,
    low: Series,
    close: Series,
    length: Optional[int] = None,
    c: Optional[float] = None,
    talib: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Commodity Channel Index (CCI)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 14
    c = float(c) if c and c > 0 else 0.015
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    if high is None or low is None or close is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from talib import CCI

        cci = CCI(high, low, close, length)
    else:
        typical_price = hlc3(high=high, low=low, close=close)
        mean_typical_price = sma(typical_price, length=length)
        mad_typical_price = mad(typical_price, length=length)

        cci = typical_price - mean_typical_price
        cci /= c * mad_typical_price

    return _finalize(cci, offset, f"CCI_{length}_{c}", "momentum", **kwargs)


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
        version. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
