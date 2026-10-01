# TTM Trend (TTM_TREND)
from typing import Any

from pandas import DataFrame, Series

from pandas_ta_classic.overlap.hl2 import hl2
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _pos_int, nan_on_short_input


@nan_on_short_input
def ttm_trend(
    high: Series,
    low: Series,
    close: Series,
    length: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> DataFrame | None:
    """Indicator: TTM Trend (TTM_TRND)"""
    # Validate arguments
    length = _pos_int(length, 6, "length")
    high = verify_series(high, length + 1)
    low = verify_series(low, length + 1)
    close = verify_series(close, length + 1)
    offset = get_offset(offset)

    if high is None or low is None or close is None:
        return None

    # Calculate Result: close against the average HL2 of the previous `length`
    # bars (the current bar is not part of the average, as in the source).
    trend_avg = hl2(high, low).shift(1).rolling(length).mean()
    # No trend until the average exists: NaN, not a -1 "downtrend".
    tm_trend = ((close > trend_avg).astype(int) * 2 - 1).where(trend_avg.notna())

    # Offset
    tm_trend = apply_offset(tm_trend, offset)

    tm_trend = apply_fill(tm_trend, **kwargs)

    # Name and Categorize it
    tm_trend.name = f"TTM_TRND_{length}"
    tm_trend.category = "momentum"

    # Prepare DataFrame to return
    data = {tm_trend.name: tm_trend}
    df = DataFrame(data)
    df.name = f"TTMTREND_{length}"
    df.category = tm_trend.category

    return df


ttm_trend.__doc__ = """TTM Trend (TTM_TRND)

This indicator is from John Carter's book “Mastering the Trade” and plots the
bars green or red. It checks if the close is above or under the average HL2 of
the `length` bars before it — the six previous bars by default, with the current
bar left out, as in the cited source. The indicator should help you stay in a
trade until the colors change. Two bars of the opposite color is the signal to
get in or out.

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
    pd.DataFrame: TTM_TRND_<length>: +1 when the close is above the average HL2
        of the previous `length` bars, -1 otherwise, NaN until that average exists.
"""
