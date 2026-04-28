# -*- coding: utf-8 -*-
# Williams Accumulation/Distribution (WAD)
from typing import Any, Optional

from pandas import Series

from pandas_ta_classic.utils import get_offset, verify_series


def wad(
    high: Series,
    low: Series,
    close: Series,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Williams Accumulation/Distribution (WAD)"""
    # Validate Arguments
    high = verify_series(high)
    low = verify_series(low)
    close = verify_series(close)
    offset = get_offset(offset)

    if high is None or low is None or close is None:
        return None

    # Calculate Result
    prev_close = close.shift(1)

    # True Range High and True Range Low
    trh = high.combine(prev_close, max)
    trl = low.combine(prev_close, min)

    # Accumulation/Distribution for each bar
    ad_day = Series(0.0, index=close.index)
    up_mask = close > prev_close
    dn_mask = close < prev_close

    ad_day[up_mask] = (close - trl)[up_mask]
    ad_day[dn_mask] = (close - trh)[dn_mask]

    wad_ = ad_day.cumsum()

    # Offset
    if offset != 0:
        wad_ = wad_.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        wad_.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        if "fill_method" in kwargs:
            if kwargs["fill_method"] == "ffill":
                wad_.ffill(inplace=True)
            elif kwargs["fill_method"] == "bfill":
                wad_.bfill(inplace=True)

    # Name and Categorize it
    wad_.name = "WAD"
    wad_.category = "volume"

    return wad_


wad.__doc__ = """Williams Accumulation/Distribution (WAD)

Williams' Accumulation/Distribution line measures the cumulative flow of
money into and out of a security. Unlike the standard A/D line, it uses
True Range High and True Range Low to determine the daily accumulation or
distribution value.

TRH = max(prev_close, high)
TRL = min(prev_close, low)

AD_day = close - TRL  if close > prev_close
AD_day = close - TRH  if close < prev_close
AD_day = 0            if close == prev_close

WAD = cumsum(AD_day)

Sources:
    Larry Williams, "How I Made One Million Dollars Last Year Trading Commodities"
    https://www.investopedia.com/terms/w/williamspctR.asp

Args:
    high (pd.Series): High price series.
    low (pd.Series): Low price series.
    close (pd.Series): Close price series.
    offset (int): Result offset. Default: 0

Returns:
    pd.Series: WAD values.
"""
