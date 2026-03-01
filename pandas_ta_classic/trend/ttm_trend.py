# -*- coding: utf-8 -*-
# TTM Trend (TTM_TREND)
from typing import Any, Optional
from pandas import DataFrame, Series
from pandas_ta_classic.overlap.hl2 import hl2
from pandas_ta_classic.utils import _build_dataframe, get_offset, verify_series


def ttm_trend(
    high: Series,
    low: Series,
    close: Series,
    length: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: TTM Trend (TTM_TRND)"""
    # Validate arguments
    length = int(length) if length and length > 0 else 6
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    offset = get_offset(offset)

    if high is None or low is None or close is None:
        return None

    # Calculate Result
    trend_avg = hl2(high, low)
    for i in range(1, length):
        trend_avg = trend_avg + hl2(high.shift(i), low.shift(i))

    trend_avg = trend_avg / length

    tm_trend = (close > trend_avg).astype(int)
    tm_trend.replace(0, -1, inplace=True)

    # Offset, Name and Categorize it
    return _build_dataframe(
        {f"TTM_TRND_{length}": tm_trend},
        f"TTMTREND_{length}",
        "momentum",
        offset,
        **kwargs,
    )


ttm_trend.__doc__ = """TTM Trend (TTM_TRND)

This indicator is from John Carters book “Mastering the Trade” and plots the
bars green or red. It checks if the price is above or under the average price of
the previous 5 bars. The indicator should hep you stay in a trade until the
colors chance. Two bars of the opposite color is the signal to get in or out.

Sources:
    https://www.prorealcode.com/prorealtime-indicators/ttm-trend-price/

Calculation:
    Default Inputs:
        length=6
    averageprice = (((high[5]+low[5])/2)+((high[4]+low[4])/2)+((high[3]+low[3])/2)+((high[2]+low[2])/2)+((high[1]+low[1])/2)+((high[6]+low[6])/2)) / 6

    if close > averageprice:
        drawcandle(open,high,low,close) coloured(0,255,0)

    if close < averageprice:
        drawcandle(open,high,low,close) coloured(255,0,0)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 6
    offset (int): How many periods to offset the result. Default: 0
Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method
Returns:
    pd.DataFrame: ttm_trend.
"""
