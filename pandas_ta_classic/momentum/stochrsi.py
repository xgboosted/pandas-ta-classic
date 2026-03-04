# Stochastic RSI (STOCHRSI)
from typing import Any, Optional
from pandas import DataFrame, Series
from .rsi import rsi
from pandas_ta_classic.overlap.ma import ma
from pandas_ta_classic.utils import (
    _build_dataframe,
    get_offset,
    non_zero_range,
    verify_series,
)


def stochrsi(
    close: Series,
    length: Optional[int] = None,
    rsi_length: Optional[int] = None,
    k: Optional[int] = None,
    d: Optional[int] = None,
    mamode: Optional[str] = None,
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

    if close is None:
        return None

    # Calculate Result
    rsi_ = rsi(close, length=rsi_length)
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

    # Offset + Name + Category + DataFrame
    _props = f"_{length}_{rsi_length}_{k}_{d}"
    return _build_dataframe(
        {f"STOCHRSIk{_props}": stochrsi_k, f"STOCHRSId{_props}": stochrsi_d},
        f"STOCHRSI{_props}",
        "momentum",
        offset,
        **kwargs,
    )


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

    RSI = RSI(close, rsi_length)
    LL  = lowest RSI for last ``length`` periods
    HH  = highest RSI for last ``length`` periods

    STOCHRSI  = 100 * (RSI - LL) / (HH - LL)
    STOCHRSIk = SMA(STOCHRSI, k)
    STOCHRSId = SMA(STOCHRSIk, d)

TA-Lib parameter mapping:
    TA-Lib STOCHRSI(timeperiod, fastk_period, fastd_period, fastd_matype)
    timeperiod  -> rsi_length  (RSI lookback)
    fastk_period -> length     (stochastic lookback for min/max)
    fastd_period -> k          (smoothing of %K)
    Note: TA-Lib's FastK is the *raw* stochastic (unsmoothed);
    this library's %K is already SMA-smoothed, so it corresponds
    to TA-Lib's FastD, not FastK.

Args:
    close (pd.Series): Series of 'close's
    length (int): The stochastic lookback period (min/max window).
        Default: 14 (TA-Lib default: fastk_period=5)
    rsi_length (int): RSI period. Default: 14
    k (int): SMA smoothing period for %K. Default: 3
    d (int): SMA smoothing period for %D. Default: 3
    mamode (str): See ```help(ta.ma)```. Default: 'sma'
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: RSI %K, RSI %D columns.
"""
