# -*- coding: utf-8 -*-
# Acceleration Bands (ACCBANDS)
from typing import Any, Optional
from pandas import DataFrame, Series
from pandas_ta_classic.overlap.ma import ma
from pandas_ta_classic.utils import (
    _build_dataframe,
    get_drift,
    get_offset,
    non_zero_range,
    verify_series,
)


def accbands(
    high: Series,
    low: Series,
    close: Series,
    length: Optional[int] = None,
    c: Optional[float] = None,
    drift: Optional[int] = None,
    mamode: Optional[str] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: Acceleration Bands (ACCBANDS)"""
    # Validate arguments
    length = int(length) if length and length > 0 else 20
    c = float(c) if c and c > 0 else 4
    mamode = mamode if isinstance(mamode, str) else "sma"
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    drift = get_drift(drift)
    offset = get_offset(offset)

    if high is None or low is None or close is None:
        return None

    # Calculate Result
    high_low_range = non_zero_range(high, low)
    hl_ratio = high_low_range / (high + low)
    hl_ratio *= c
    _lower = low * (1 - hl_ratio)
    _upper = high * (1 + hl_ratio)

    lower = ma(mamode, _lower, length=length)
    mid = ma(mamode, close, length=length)
    upper = ma(mamode, _upper, length=length)
    if lower is None or mid is None or upper is None:
        return None

    _props = f"_{length}"
    return _build_dataframe(
        {f"ACCBL{_props}": lower, f"ACCBM{_props}": mid, f"ACCBU{_props}": upper},
        f"ACCBANDS{_props}",
        "volatility",
        offset,
        **kwargs,
    )


accbands.__doc__ = """Acceleration Bands (ACCBANDS)

Acceleration Bands created by Price Headley plots upper and lower envelope
bands around a simple moving average.

Sources:
    https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/acceleration-bands-abands/

Calculation:
    Default Inputs:
        length=10, c=4
    EMA = Exponential Moving Average
    SMA = Simple Moving Average
    HL_RATIO = c * (high - low) / (high + low)
    LOW = low * (1 - HL_RATIO)
    HIGH = high * (1 + HL_RATIO)

    if 'ema':
        LOWER = EMA(LOW, length)
        MID = EMA(close, length)
        UPPER = EMA(HIGH, length)
    else:
        LOWER = SMA(LOW, length)
        MID = SMA(close, length)
        UPPER = SMA(HIGH, length)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 10
    c (int): Multiplier. Default: 4
    mamode (str): See ```help(ta.ma)```. Default: 'sma'
    drift (int): The difference period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: lower, mid, upper columns.
"""
