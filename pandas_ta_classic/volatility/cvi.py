# -*- coding: utf-8 -*-
# Chaikins Volatility (CVI)
from typing import Any, Optional

from pandas import Series

from pandas_ta_classic.overlap.ema import ema
from pandas_ta_classic.utils import get_offset, verify_series


def cvi(
    high: Series,
    low: Series,
    length: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Chaikins Volatility (CVI)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 10
    high = verify_series(high, length)
    low = verify_series(low, length)
    offset = get_offset(offset)

    if high is None or low is None:
        return None

    # Calculate Result
    hl = high - low
    ema_hl = ema(hl, length=length, talib=False)
    if ema_hl is None:
        return None

    cvi_ = 100 * (ema_hl - ema_hl.shift(length)) / ema_hl.shift(length)

    # Offset
    if offset != 0:
        cvi_ = cvi_.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        cvi_.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        if "fill_method" in kwargs:
            if kwargs["fill_method"] == "ffill":
                cvi_.ffill(inplace=True)
            elif kwargs["fill_method"] == "bfill":
                cvi_.bfill(inplace=True)

    # Name and Categorize it
    cvi_.name = f"CVI_{length}"
    cvi_.category = "volatility"

    return cvi_


cvi.__doc__ = """Chaikins Volatility (CVI)

Chaikins Volatility measures the range between the high and low prices by
calculating the rate of change of the exponential moving average of the
High-Low spread. Rising CVI indicates expanding volatility; falling CVI
indicates contracting volatility.

HL = High - Low
EMA_HL = EMA(HL, length)
CVI = 100 * (EMA_HL - EMA_HL[length]) / EMA_HL[length]

Sources:
    Marc Chaikin
    https://school.stockcharts.com/doku.php?id=technical_indicators:chaikins_volatility

Args:
    high (pd.Series): High price series.
    low (pd.Series): Low price series.
    length (int): EMA period and lookback. Default: 10
    offset (int): Result offset. Default: 0

Returns:
    pd.Series: CVI values.
"""
