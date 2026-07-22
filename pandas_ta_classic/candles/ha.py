# Heikin Ashi (HA)
from typing import Any, Optional
import numpy as np
from pandas import DataFrame, Series
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._njit import njit


@njit(cache=True)
def _ha_open(first: float, ha_close: np.ndarray) -> np.ndarray:
    """Recursive Heikin-Ashi open: HA_open[i] = (HA_open[i-1] + HA_close[i-1]) / 2."""
    m = ha_close.shape[0]
    out = np.empty(m)
    out[0] = first
    for i in range(1, m):
        out[i] = 0.5 * (out[i - 1] + ha_close[i - 1])
    return out


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
    ha_close = 0.25 * (open_ + high + low + close)
    ha_open = _ha_open(0.5 * (open_.iloc[0] + close.iloc[0]), ha_close.to_numpy(dtype=float))
    df = DataFrame(
        {
            "HA_open": ha_open,
            "HA_high": high,
            "HA_low": low,
            "HA_close": ha_close,
        },
        index=close.index,
    )

    df["HA_high"] = df[["HA_open", "HA_high", "HA_close"]].max(axis=1)
    df["HA_low"] = df[["HA_open", "HA_low", "HA_close"]].min(axis=1)

    # Offset
    df = apply_offset(df, offset)

    df = apply_fill(df, **kwargs)

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
