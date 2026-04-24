# -*- coding: utf-8 -*-
# Chandelier Exit (CE)
from typing import Any, Optional
from pandas import DataFrame, Series
from .atr import atr
from pandas_ta_classic.utils import get_offset, verify_series


def ce(
    high: Series,
    low: Series,
    close: Series,
    length: Optional[int] = None,
    multiplier: Optional[float] = None,
    mamode: Optional[str] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: Chandelier Exit (CE)"""
    # Validate arguments
    length = int(length) if length and length > 0 else 22
    multiplier = float(multiplier) if multiplier and multiplier > 0 else 3.0

    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    offset = get_offset(offset)

    if high is None or low is None or close is None:
        return None

    # Calculate Result
    atr_ = atr(high=high, low=low, close=close, length=length, mamode=mamode)
    if atr_ is None:
        return None

    highest_high = high.rolling(length, min_periods=length).max()
    lowest_low = low.rolling(length, min_periods=length).min()

    ce_long = highest_high - (atr_ * multiplier)
    ce_short = lowest_low + (atr_ * multiplier)

    data = {
        f"CE_L_{length}_{multiplier}": ce_long,
        f"CE_S_{length}_{multiplier}": ce_short,
    }
    df = DataFrame(data, index=close.index)

    # Offset
    if offset != 0:
        df = df.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        df.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        if kwargs["fill_method"] == "ffill":
            df.ffill(inplace=True)
        elif kwargs["fill_method"] == "bfill":
            df.bfill(inplace=True)

    # Name and Categorize it
    df.name = f"CE_{length}_{multiplier}"
    df.category = "volatility"

    return df


ce.__doc__ = """Chandelier Exit (CE)

The Chandelier Exit is a volatility-based trailing stop indicator that uses the
Average True Range (ATR) to determine exit levels. It helps traders stay in a
trend while providing dynamic stop-loss levels based on market volatility.

Sources:
    https://www.tradingview.com/support/solutions/43000773013-chandelier-exit/
    https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/chandelier-exit

Calculation:
    Default Inputs:
        length=22, multiplier=3.0
    ATR = Average True Range(length)
    Highest High = rolling max(high, length)
    Lowest Low = rolling min(low, length)

    CE Long = Highest High - (Multiplier * ATR)
    CE Short = Lowest Low + (Multiplier * ATR)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): Lookback period for High/Low and ATR. Default: 22
    multiplier (float): ATR multiplier. Default: 3.0
    mamode (str): See ``help(ta.ma)``. Default: 'rma'
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method ('ffill' or 'bfill')

Returns:
    pd.DataFrame: CE_L (Long) and CE_S (Short) columns.

Examples:
    >>> import pandas as pd
    >>> import pandas_ta_classic as ta
    >>> df = pd.DataFrame({
    ...     "high":  [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    ...     "low":   [5,  6,  7,  8,  9,  10, 11, 12, 13, 14],
    ...     "close": [7,  8,  9,  10, 11, 12, 13, 14, 15, 16],
    ... })
    >>> result = ta.ce(high=df["high"], low=df["low"], close=df["close"], length=5)
    >>> result.tail()
"""
