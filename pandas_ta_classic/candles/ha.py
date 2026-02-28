# -*- coding: utf-8 -*-
# Heikin Ashi (HA)
from typing import Any, Optional
from pandas import DataFrame, Series
from pandas_ta_classic.utils import get_offset, verify_series


def ha(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: Candle Type - Heikin Ashi"""
    # Validate Arguments
    open_ = verify_series(open_)
    high = verify_series(high)
    low = verify_series(low)
    close = verify_series(close)
    offset = get_offset(offset)

    if open_ is None or high is None or low is None or close is None:
        return None

    # Calculate Result
    import numpy as np

    m = close.size
    ha_close = 0.25 * (
        open_.to_numpy() + high.to_numpy() + low.to_numpy() + close.to_numpy()
    )

    # HA_open recurrence: ha_open[i] = 0.5 * (ha_open[i-1] + ha_close[i-1])
    # Compute with a numpy loop to avoid pandas iat overhead.
    ha_open = np.empty(m)
    ha_open[0] = 0.5 * (open_.iloc[0] + close.iloc[0])
    for i in range(1, m):
        ha_open[i] = 0.5 * (ha_open[i - 1] + ha_close[i - 1])

    ha_high = np.maximum(np.maximum(ha_open, high.to_numpy()), ha_close)
    ha_low = np.minimum(np.minimum(ha_open, low.to_numpy()), ha_close)

    df = DataFrame(
        {
            "HA_open": ha_open,
            "HA_high": ha_high,
            "HA_low": ha_low,
            "HA_close": ha_close,
        },
        index=close.index,
    )

    # Offset
    if offset != 0:
        df = df.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        df.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        if "fill_method" in kwargs:

            if kwargs["fill_method"] == "ffill":

                df.ffill(inplace=True)

            elif kwargs["fill_method"] == "bfill":

                df.bfill(inplace=True)

    # Name and Categorize it
    df.name = "Heikin-Ashi"
    df.category = "candles"

    return df


ha.__doc__ = """Heikin Ashi Candles (HA)

The Heikin-Ashi technique averages price data to create a Japanese
candlestick chart that filters out market noise. Heikin-Ashi charts,
developed by Munehisa Homma in the 1700s, share some characteristics
with standard candlestick charts but differ based on the values used
to create each candle. Instead of using the open, high, low, and close
like standard candlestick charts, the Heikin-Ashi technique uses a
modified formula based on two-period averages. This gives the chart a
smoother appearance, making it easier to spots trends and reversals,
but also obscures gaps and some price data.

Sources:
    https://www.investopedia.com/terms/h/heikinashi.asp

Calculation:
    HA_OPEN[0] = (open[0] + close[0]) / 2
    HA_CLOSE = (open[0] + high[0] + low[0] + close[0]) / 4

    for i > 1 in df.index:
        HA_OPEN = (HA_OPEN[i−1] + HA_CLOSE[i−1]) / 2

    HA_HIGH = MAX(HA_OPEN, HA_HIGH, HA_CLOSE)
    HA_LOW = MIN(HA_OPEN, HA_LOW, HA_CLOSE)

    How to Calculate Heikin-Ashi

    Use one period to create the first Heikin-Ashi (HA) candle, using
    the formulas. For example use the high, low, open, and close to
    create the first HA close price. Use the open and close to create
    the first HA open. The high of the period will be the first HA high,
    and the low will be the first HA low. With the first HA calculated,
    it is now possible to continue computing the HA candles per the formulas.
​​
Args:
    open_ (pd.Series): Series of 'open's
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: ha_open, ha_high,ha_low, ha_close columns.
"""
