# Average True Range (ATR)
from typing import Annotated, Any

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
from pandas_ta_classic.utils._core import Bound, Flag, OneOf, Options, nan_on_short_input

from .true_range import true_range


@nan_on_short_input
def atr(
    high: Series,
    low: Series,
    close: Series,
    length: Annotated[int, Bound(gt=0)] = 14,
    mamode: Annotated[str, OneOf(tuple(ma()))] = 'rma',
    talib: Annotated[bool, Flag()] = False,
    drift: int | None = None,
    offset: int | None = None,
    **kwargs: Annotated[Any, Options(percent=Flag(False))],
) -> Series | None:
    """Indicator: Average True Range (ATR)"""
    # Validate arguments
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    drift = get_drift(drift)
    offset = get_offset(offset)

    if high is None or low is None or close is None:
        return None

    # Calculate Result
    # TA-Lib cannot express a non-default drift; run natively instead of ignoring it
    if Imports["talib"] and talib and mamode == "rma" and drift == 1:
        from talib import ATR

        atr = ATR(high, low, close, length)
    else:
        tr = true_range(high=high, low=low, close=close, drift=drift)
        atr = ma(mamode, tr, length=length)
        if atr is None:
            return None

    percentage = kwargs.pop("percent")
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
