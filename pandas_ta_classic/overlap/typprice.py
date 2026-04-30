# -*- coding: utf-8 -*-
# Typical Price (TYPPRICE)
from typing import Any, Optional
from pandas import Series
from pandas_ta_classic.utils import get_offset, verify_series


def typprice(
    high: Series,
    low: Series,
    close: Series,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Typical Price (TYPPRICE)

    Per-bar (High + Low + Close) / 3.  Equivalent to ta.hlc3.
    TA-Lib name: TYPPRICE.
    """
    high = verify_series(high)
    low = verify_series(low)
    close = verify_series(close)
    offset = get_offset(offset)

    if high is None or low is None or close is None:
        return None

    result = (high + low + close) / 3.0

    if offset != 0:
        result = result.shift(offset)

    result.name = "TYPPRICE"
    result.category = "overlap"
    return result


typprice.__doc__ = """Typical Price (TYPPRICE)

TYPPRICE = (High + Low + Close) / 3

Equivalent to ta.hlc3.  TA-Lib name: TYPPRICE.

Args:
    high (pd.Series): Series of 'high' prices
    low (pd.Series): Series of 'low' prices
    close (pd.Series): Series of 'close' prices
    offset (int): Periods to offset. Default: 0

Returns:
    pd.Series
"""
