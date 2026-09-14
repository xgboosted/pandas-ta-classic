# Stochastic Oscillator (STOCH)
from typing import Any

from pandas import DataFrame, Series

from pandas_ta_classic import Imports
from pandas_ta_classic.overlap.ma import ma
from pandas_ta_classic.utils import (
    apply_fill,
    apply_offset,
    get_offset,
    non_zero_range,
    verify_series,
)
from pandas_ta_classic.utils._core import _bool_param, _pos_int, _str_param, nan_on_short_input


@nan_on_short_input
def stoch(
    high: Series,
    low: Series,
    close: Series,
    k: int | None = None,
    d: int | None = None,
    smooth_k: int | None = None,
    mamode: str | None = None,
    talib: bool | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> DataFrame | None:
    """Indicator: Stochastic Oscillator (STOCH)"""
    # Validate arguments
    k = _pos_int(k, 14, "k")
    d = _pos_int(d, 3, "d")
    smooth_k = _pos_int(smooth_k, 3, "smooth_k")
    _length = max(k, d, smooth_k)
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    close = verify_series(close, _length)
    offset = get_offset(offset)
    mamode = _str_param(mamode, "sma", "mamode")
    mode_talib = _bool_param(talib, False, "talib")

    if high is None or low is None or close is None:
        return None

    # Calculate Result
    # TA-Lib cannot express a non-default mamode; run natively instead of ignoring it
    if Imports["talib"] and mode_talib and mamode == "sma":
        from talib import STOCH

        _k, _d = STOCH(
            high,
            low,
            close,
            fastk_period=k,
            slowk_period=smooth_k,
            slowk_matype=0,
            slowd_period=d,
            slowd_matype=0,
        )
        stoch_k = Series(_k, index=close.index)
        stoch_d = Series(_d, index=close.index)
    else:
        lowest_low = low.rolling(k).min()
        highest_high = high.rolling(k).max()

        stoch = 100 * (close - lowest_low)
        stoch /= non_zero_range(highest_high, lowest_low)

        stoch_k = ma(mamode, stoch.loc[stoch.first_valid_index() :,], length=smooth_k)
        if stoch_k is None:
            return None
        stoch_d = ma(mamode, stoch_k.loc[stoch_k.first_valid_index() :,], length=d)
        if stoch_d is None:
            return None

        # The warmup slices above shorten %K and %D; restore the caller's index
        # so the result lines up bar for bar with close.
        stoch_k = stoch_k.reindex(close.index)
        stoch_d = stoch_d.reindex(close.index)

    # Offset
    stoch_k, stoch_d = apply_offset([stoch_k, stoch_d], offset)

    stoch_k, stoch_d = apply_fill([stoch_k, stoch_d], **kwargs)

    # Name and Categorize it
    _name = "STOCH"
    _props = f"_{k}_{d}_{smooth_k}"
    stoch_k.name = f"{_name}k{_props}"
    stoch_d.name = f"{_name}d{_props}"
    stoch_k.category = stoch_d.category = "momentum"

    # Prepare DataFrame to return
    data = {stoch_k.name: stoch_k, stoch_d.name: stoch_d}
    df = DataFrame(data)
    df.name = f"{_name}{_props}"
    df.category = stoch_k.category
    return df


stoch.__doc__ = """Stochastic (STOCH)

The Stochastic Oscillator (STOCH) was developed by George Lane in the 1950's.
He believed this indicator was a good way to measure momentum because changes in
momentum precede changes in price.

It is a range-bound oscillator with two lines moving between 0 and 100.
The first line (%K) displays the current close in relation to the period's
high/low range. The second line (%D) is a Simple Moving Average of the %K line.
The most common choices are a 14 period %K and a 3 period SMA for %D.

Sources:
    https://www.tradingview.com/wiki/Stochastic_(STOCH)
    https://www.sierrachart.com/index.php?page=doc/StudiesReference.php&ID=332&Name=KD_-_Slow

Calculation:
    Default Inputs:
        k=14, d=3, smooth_k=3
    SMA = Simple Moving Average
    LL  = low for last k periods
    HH  = high for last k periods

    STOCH = 100 * (close - LL) / (HH - LL)
    STOCHk = SMA(STOCH, smooth_k)
    STOCHd = SMA(FASTK, d)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    k (int): The Fast %K period. Default: 14
    d (int): The Slow %K period. Default: 3
    smooth_k (int): The Slow %D period. Default: 3
    mamode (str): See ```help(ta.ma)```. Default: 'sma'
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version (SMA smoothing only). Default: False
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: %K, %D columns.
"""
