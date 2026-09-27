# Market Facilitation Index (MARKETFI)
from typing import Any

from pandas import Series

from pandas_ta_classic.utils import (
    apply_fill,
    apply_offset,
    get_offset,
    verify_series,
)
from pandas_ta_classic.utils._core import nan_on_short_input


@nan_on_short_input
def marketfi(
    high: Series,
    low: Series,
    volume: Series,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Market Facilitation Index (MARKETFI)"""
    # Validate Arguments
    high = verify_series(high)
    low = verify_series(low)
    volume = verify_series(volume)
    offset = get_offset(offset)

    if high is None or low is None or volume is None:
        return None

    # Calculate Result
    # The range is the numerator, so non_zero_range() only ever reported
    # movement a flat bar did not make: eps / volume. A bar that traded at one
    # price on no volume divides 0/0 and reads 0.0, the convention TA-Lib
    # applies for a degenerate window; see tests/test_degenerate_input.py. A
    # real range on no volume is still a division by zero, as it was.
    high_low_range = high - low
    marketfi_ = (high_low_range / volume).mask((volume == 0) & (high_low_range == 0), 0.0)

    # Offset
    marketfi_ = apply_offset(marketfi_, offset)

    marketfi_ = apply_fill(marketfi_, **kwargs)

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

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: MARKETFI values.
"""
