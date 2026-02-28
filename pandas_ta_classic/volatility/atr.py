# -*- coding: utf-8 -*-
# Average True Range (ATR)
from typing import Any, Optional
from pandas import Series
from .true_range import true_range
from pandas_ta_classic import Imports
from pandas_ta_classic.overlap.ma import ma
from pandas_ta_classic.utils import apply_offset, get_drift, get_offset, verify_series


def atr(
    high: Series,
    low: Series,
    close: Series,
    length: Optional[int] = None,
    mamode: Optional[str] = None,
    talib: Optional[bool] = None,
    drift: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Average True Range (ATR)"""
    # Validate arguments
    length = int(length) if length and length > 0 else 14
    mamode = mamode.lower() if mamode and isinstance(mamode, str) else "rma"
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    drift = get_drift(drift)
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    if high is None or low is None or close is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from talib import ATR

        atr = ATR(high, low, close, length)
    else:
        tr = true_range(high=high, low=low, close=close, drift=drift)
        atr = ma(mamode, tr, length=length)

    percentage = kwargs.pop("percent", False)
    if percentage:
        atr *= 100 / close

    # Offset
    atr = apply_offset(atr, offset, **kwargs)

    # Name and Categorize it
    atr.name = f"ATR{mamode[0]}_{length}{'p' if percentage else ''}"
    atr.category = "volatility"

    return atr


atr.__doc__ = """Average True Range (ATR)

Averge True Range is used to measure volatility, especially volatility caused by
gaps or limit moves.

Sources:
    https://www.tradingview.com/wiki/Average_True_Range_(ATR)

Calculation:
    Default Inputs:
        length=14, drift=1, percent=False
    EMA = Exponential Moving Average
    SMA = Simple Moving Average
    WMA = Weighted Moving Average
    RMA = WildeR's Moving Average
    TR = True Range

    tr = TR(high, low, close, drift)
    if 'ema':
        ATR = EMA(tr, length)
    elif 'sma':
        ATR = SMA(tr, length)
    elif 'wma':
        ATR = WMA(tr, length)
    else:
        ATR = RMA(tr, length)

    if percent:
        ATR *= 100 / close

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 14
    mamode (str): See ```help(ta.ma)```. Default: 'rma'
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    drift (int): The difference period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    percent (bool, optional): Return as percentage. Default: False
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
