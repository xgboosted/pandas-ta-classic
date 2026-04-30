# -*- coding: utf-8 -*-
# Market Facilitation Index (MARKETFI)
from typing import Any, Optional

from pandas import Series

from pandas_ta_classic.utils import get_offset, non_zero_range, verify_series


def marketfi(
    high: Series,
    low: Series,
    volume: Series,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Market Facilitation Index (MARKETFI)"""
    # Validate Arguments
    high = verify_series(high)
    low = verify_series(low)
    volume = verify_series(volume)
    offset = get_offset(offset)

    if high is None or low is None or volume is None:
        return None

    # Calculate Result
    marketfi_ = non_zero_range(high, low) / volume

    # Offset
    if offset != 0:
        marketfi_ = marketfi_.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        marketfi_.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        if "fill_method" in kwargs:
            if kwargs["fill_method"] == "ffill":
                marketfi_.ffill(inplace=True)
            elif kwargs["fill_method"] == "bfill":
                marketfi_.bfill(inplace=True)

    # Name and Categorize it
    marketfi_.name = "MARKETFI"
    marketfi_.category = "volume"

    return marketfi_


marketfi.__doc__ = """Market Facilitation Index (MARKETFI)

The Market Facilitation Index (MFI) was developed by Dr. Bill Williams.
It measures the efficiency of price movement per unit of volume. A higher
value means each tick of volume is driving more price movement.

MARKETFI = (High - Low) / Volume

Sources:
    Bill Williams, "Trading Chaos", 1995
    https://www.investopedia.com/terms/m/marketfacilitationindex.asp

Args:
    high (pd.Series): High price series.
    low (pd.Series): Low price series.
    volume (pd.Series): Volume series.
    offset (int): Result offset. Default: 0

Returns:
    pd.Series: MARKETFI values.
"""
