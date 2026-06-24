# Stochastic RSI (STOCHRSI)
from typing import Any, Optional
from pandas import DataFrame, Series
from .rsi import rsi
from pandas_ta_classic import Imports
from pandas_ta_classic.overlap.ma import ma
from pandas_ta_classic.utils import (
    apply_fill,
    apply_offset,
    get_offset,
    non_zero_range,
    verify_series,
)


def _stochrsi_result_df(k_series, d_series, length, rsi_length, k, d):
    """Attach names/categories and return a result :class:`~pandas.DataFrame`.

    Extracted to avoid repeating the identical naming block in both the
    TA-Lib and the pure-Python code paths.

    Args:
        k_series (Series): Stochastic %K series.
        d_series (Series): Stochastic %D series.
        length (int): STOCHRSI look-back period.
        rsi_length (int): RSI look-back period.
        k (int): %K smoothing period.
        d (int): %D smoothing period.

    Returns:
        DataFrame: Two columns ``STOCHRSIk_…`` and ``STOCHRSId_…``.
    """
    _name = "STOCHRSI"
    _props = f"_{length}_{rsi_length}_{k}_{d}"
    k_series.name = f"{_name}k{_props}"
    d_series.name = f"{_name}d{_props}"
    k_series.category = d_series.category = "momentum"
    data = {k_series.name: k_series, d_series.name: d_series}
    df = DataFrame(data)
    df.name = f"{_name}{_props}"
    df.category = k_series.category
    return df


def stochrsi(
    close: Series,
    length: Optional[int] = None,
    rsi_length: Optional[int] = None,
    k: Optional[int] = None,
    d: Optional[int] = None,
    mamode: Optional[str] = None,
    talib: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: Stochastic RSI Oscillator (STOCHRSI)"""
    # Validate arguments
    length = length if length and length > 0 else 14
    rsi_length = rsi_length if rsi_length and rsi_length > 0 else 14
    k = k if k and k > 0 else 3
    d = d if d and d > 0 else 3
    close = verify_series(close, max(length, rsi_length, k, d))
    offset = get_offset(offset)
    mamode = mamode if isinstance(mamode, str) else "sma"
    mode_talib = bool(talib) if isinstance(talib, bool) else False

    if close is None:
        return None

    if Imports["talib"] and mode_talib:
        from talib import STOCHRSI as _STOCHRSI

        fastk, fastd = _STOCHRSI(close, timeperiod=length, fastk_period=length, fastd_period=d)
        stochrsi_k = Series(fastk, index=close.index)
        stochrsi_d = Series(fastd, index=close.index)
        stochrsi_k, stochrsi_d = apply_offset([stochrsi_k, stochrsi_d], offset)
        return _stochrsi_result_df(stochrsi_k, stochrsi_d, length, rsi_length, k, d)

    # Calculate Result
    rsi_ = rsi(close, length=rsi_length)
    if rsi_ is None:
        return None
    lowest_rsi = rsi_.rolling(length).min()
    highest_rsi = rsi_.rolling(length).max()

    stoch = 100 * (rsi_ - lowest_rsi)
    stoch /= non_zero_range(highest_rsi, lowest_rsi)

    stochrsi_k = ma(mamode, stoch, length=k)
    if stochrsi_k is None:
        return None
    stochrsi_d = ma(mamode, stochrsi_k, length=d)
    if stochrsi_d is None:
        return None

    # Offset
    stochrsi_k, stochrsi_d = apply_offset([stochrsi_k, stochrsi_d], offset)

    stochrsi_k, stochrsi_d = apply_fill([stochrsi_k, stochrsi_d], **kwargs)

    return _stochrsi_result_df(stochrsi_k, stochrsi_d, length, rsi_length, k, d)


stochrsi.__doc__ = """Stochastic (STOCHRSI)

"Stochastic RSI and Dynamic Momentum Index" was created by Tushar Chande and Stanley Kroll and published in Stock & Commodities V.11:5 (189-199)

It is a range-bound oscillator with two lines moving between 0 and 100.
The first line (%K) displays the current RSI in relation to the period's
high/low range. The second line (%D) is a Simple Moving Average of the %K line.
The most common choices are a 14 period %K and a 3 period SMA for %D.

Sources:
    https://www.tradingview.com/wiki/Stochastic_(STOCH)

Calculation:
    Default Inputs:
        length=14, rsi_length=14, k=3, d=3
    RSI = Relative Strength Index
    SMA = Simple Moving Average

    RSI = RSI(high, low, close, rsi_length)
    LL  = lowest RSI for last rsi_length periods
    HH  = highest RSI for last rsi_length periods

    STOCHRSI  = 100 * (RSI - LL) / (HH - LL)
    STOCHRSIk = SMA(STOCHRSI, k)
    STOCHRSId = SMA(STOCHRSIk, d)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): The STOCHRSI period. Default: 14
    rsi_length (int): RSI period. Default: 14
    k (int): The Fast %K period. Default: 3
    d (int): The Slow %K period. Default: 3
    mamode (str): See ```help(ta.ma)```. Default: 'sma'
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version (rsi_length ignored). Default: False
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: RSI %K, RSI %D columns.
"""
