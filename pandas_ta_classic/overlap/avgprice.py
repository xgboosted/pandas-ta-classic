# -*- coding: utf-8 -*-
# Average Price (AVGPRICE)
from typing import Any, Optional
from pandas import Series
from pandas_ta_classic.utils import apply_offset, get_offset, verify_series


def avgprice(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Average Price (AVGPRICE)

    Equal to (Open + High + Low + Close) / 4.  Alias for ohlc4.
    TA-Lib name: AVGPRICE.
    """
    open_ = verify_series(open_)
    high = verify_series(high)
    low = verify_series(low)
    close = verify_series(close)
    offset = get_offset(offset)

    if open_ is None or high is None or low is None or close is None:
        return None

    result = 0.25 * (open_ + high + low + close)

    result = apply_offset(result, offset)

    result.name = "AVGPRICE"
    result.category = "overlap"
    return result


avgprice.__doc__ = """Average Price (AVGPRICE)

AVGPRICE = (Open + High + Low + Close) / 4

Equivalent to ta.ohlc4.  TA-Lib name: AVGPRICE.

Args:
    open_ (pd.Series): Series of 'open' prices
    high (pd.Series): Series of 'high' prices
    low (pd.Series): Series of 'low' prices
    close (pd.Series): Series of 'close' prices
    offset (int): Periods to offset. Default: 0

Returns:
    pd.Series
"""
