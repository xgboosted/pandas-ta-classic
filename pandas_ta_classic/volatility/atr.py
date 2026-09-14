# Average True Range (ATR)
from typing import Any

from pandas import Series

from pandas_ta_classic import Imports
from pandas_ta_classic.overlap.ma import ma
from pandas_ta_classic.utils import (
    apply_fill,
    apply_offset,
    get_drift,
    get_offset,
    verify_series,
)
from pandas_ta_classic.utils._core import _bool_param, _pos_int, _str_param, nan_on_short_input

from .true_range import true_range


@nan_on_short_input
def atr(
    high: Series,
    low: Series,
    close: Series,
    length: int | None = None,
    mamode: str | None = None,
    talib: bool | None = None,
    drift: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Average True Range (ATR)"""
    # Validate arguments
    length = _pos_int(length, 14, "length")
    mamode = _str_param(mamode, "rma", "mamode")
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    drift = get_drift(drift)
    offset = get_offset(offset)
    mode_talib = _bool_param(talib, False, "talib")

    if high is None or low is None or close is None:
        return None

    # Calculate Result
    # TA-Lib cannot express a non-default drift; run natively instead of ignoring it
    if Imports["talib"] and mode_talib and drift == 1:
        from talib import ATR

        atr = ATR(high, low, close, length)
    else:
        tr = true_range(high=high, low=low, close=close, drift=drift)
        atr = ma(mamode, tr, length=length)
        if atr is None:
            return None

    percentage = kwargs.pop("percent", False)
    if percentage:
        atr *= 100 / close

    # Offset
    atr = apply_offset(atr, offset)

    atr = apply_fill(atr, **kwargs)

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
        version. Default: False
    drift (int): The difference period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    percent (bool, optional): Return as percentage. Default: False
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
