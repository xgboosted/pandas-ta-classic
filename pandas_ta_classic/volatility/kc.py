# -*- coding: utf-8 -*-
# Keltner Channels (KC)
from typing import Any, Optional
from pandas import DataFrame, Series
from .true_range import true_range
from pandas_ta_classic.overlap.ma import ma
from pandas_ta_classic.utils import (
    _build_dataframe,
    get_offset,
    high_low_range,
    verify_series,
)


def kc(
    high: Series,
    low: Series,
    close: Series,
    length: Optional[int] = None,
    scalar: Optional[float] = None,
    mamode: Optional[str] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: Keltner Channels (KC)"""
    # Validate arguments
    length = int(length) if length and length > 0 else 20
    scalar = float(scalar) if scalar and scalar > 0 else 2
    mamode = mamode if isinstance(mamode, str) else "ema"
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    offset = get_offset(offset)

    if high is None or low is None or close is None:
        return None

    # Calculate Result
    use_tr = kwargs.pop("tr", True)
    if use_tr:
        range_ = true_range(high, low, close)
    else:
        range_ = high_low_range(high, low)

    basis = ma(mamode, close, length=length)
    band = ma(mamode, range_, length=length)
    if basis is None or band is None:
        return None

    lower = basis - scalar * band
    upper = basis + scalar * band

    _props = f"{mamode.lower()[0] if len(mamode) else ''}_{length}_{scalar}"
    return _build_dataframe(
        {f"KCL{_props}": lower, f"KCB{_props}": basis, f"KCU{_props}": upper},
        f"KC{_props}",
        "volatility",
        offset,
        **kwargs,
    )


kc.__doc__ = """Keltner Channels (KC)

A popular volatility indicator similar to Bollinger Bands and
Donchian Channels.

Sources:
    https://www.tradingview.com/wiki/Keltner_Channels_(KC)

Calculation:
    Default Inputs:
        length=20, scalar=2, mamode=None, tr=True
    TR = True Range
    SMA = Simple Moving Average
    EMA = Exponential Moving Average

    if tr:
        RANGE = TR(high, low, close)
    else:
        RANGE = high - low

    if mamode == "ema":
        BASIS = sma(close, length)
        BAND = sma(RANGE, length)
    elif mamode == "sma":
        BASIS = sma(close, length)
        BAND = sma(RANGE, length)

    LOWER = BASIS - scalar * BAND
    UPPER = BASIS + scalar * BAND

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): The short period.  Default: 20
    scalar (float): A positive float to scale the bands. Default: 2
    mamode (str): See ```help(ta.ma)```. Default: 'ema'
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    tr (bool): When True, it uses True Range for calculation. When False, use a
        high - low as it's range calculation. Default: True
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: lower, basis, upper columns.
"""
