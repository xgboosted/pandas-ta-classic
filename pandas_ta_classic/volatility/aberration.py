# -*- coding: utf-8 -*-
# from numpy import sqrt as npsqrt
from typing import Any, Optional
from pandas import DataFrame, Series
from .atr import atr
from pandas_ta_classic.overlap.hlc3 import hlc3
from pandas_ta_classic.overlap.sma import sma
from pandas_ta_classic.utils import apply_offset, get_offset, verify_series


def aberration(
    high: Series,
    low: Series,
    close: Series,
    length: Optional[int] = None,
    atr_length: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: Aberration (ABER)"""
    # Validate arguments
    length = int(length) if length and length > 0 else 5
    atr_length = int(atr_length) if atr_length and atr_length > 0 else 15
    _length = max(atr_length, length)
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    close = verify_series(close, _length)
    offset = get_offset(offset)

    if high is None or low is None or close is None:
        return None

    # Calculate Result
    atr_ = atr(high=high, low=low, close=close, length=atr_length)
    jg = hlc3(high=high, low=low, close=close)

    zg = sma(jg, length)
    sg = zg + atr_
    xg = zg - atr_

    # Offset
    zg = apply_offset(zg, offset, **kwargs)
    sg = apply_offset(sg, offset, **kwargs)
    xg = apply_offset(xg, offset, **kwargs)
    atr_ = apply_offset(atr_, offset, **kwargs)

    # Name and Categorize it
    _props = f"_{length}_{atr_length}"
    zg.name = f"ABER_ZG{_props}"
    sg.name = f"ABER_SG{_props}"
    xg.name = f"ABER_XG{_props}"
    atr_.name = f"ABER_ATR{_props}"
    zg.category = sg.category = "volatility"
    xg.category = atr_.category = zg.category

    # Prepare DataFrame to return
    data = {zg.name: zg, sg.name: sg, xg.name: xg, atr_.name: atr_}
    aberdf = DataFrame(data)
    aberdf.name = f"ABER{_props}"
    aberdf.category = zg.category

    return aberdf


aberration.__doc__ = """Aberration

A volatility indicator similar to Keltner Channels.

Sources:
    Few internet resources on definitive definition.
    Request by Github user homily, issue #46

Calculation:
    Default Inputs:
        length=5, atr_length=15
    ATR = Average True Range
    SMA = Simple Moving Average

    ATR = ATR(length=atr_length)
    JG = TP = HLC3(high, low, close)
    ZG = SMA(JG, length)
    SG = ZG + ATR
    XG = ZG - ATR

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): The short period. Default: 5
    atr_length (int): The short period. Default: 15
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: zg, sg, xg, atr columns.
"""
