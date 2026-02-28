# -*- coding: utf-8 -*-
# Pretty Good Oscillator (PGO)
from typing import Any, Optional
from pandas import Series
from pandas_ta_classic.overlap.ema import ema
from pandas_ta_classic.overlap.sma import sma
from pandas_ta_classic.volatility import atr
from pandas_ta_classic.utils import apply_offset, get_offset, verify_series


def pgo(
    high: Series,
    low: Series,
    close: Series,
    length: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Pretty Good Oscillator (PGO)"""
    # Validate arguments
    length = int(length) if length and length > 0 else 14
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    offset = get_offset(offset)

    if high is None or low is None or close is None:
        return None

    # Calculate Result
    pgo = close - sma(close, length)
    pgo /= ema(atr(high, low, close, length), length)

    # Offset
    pgo = apply_offset(pgo, offset, **kwargs)

    # Name and Categorize it
    pgo.name = f"PGO_{length}"
    pgo.category = "momentum"

    return pgo


pgo.__doc__ = """Pretty Good Oscillator (PGO)

The Pretty Good Oscillator indicator was created by Mark Johnson to measure the distance of the current close from its N-day Simple Moving Average, expressed in terms of an average true range over a similar period. Johnson's approach was to
use it as a breakout system for longer term trades. Long if greater than 3.0 and
short if less than -3.0.

Sources:
    https://library.tradingtechnologies.com/trade/chrt-ti-pretty-good-oscillator.html

Calculation:
    Default Inputs:
        length=14
    ATR = Average True Range
    SMA = Simple Moving Average
    EMA = Exponential Moving Average

    PGO = (close - SMA(close, length)) / EMA(ATR(high, low, close, length), length)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 14
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
