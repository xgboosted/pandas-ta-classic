# -*- coding: utf-8 -*-
# Mean Deviation (MD)
from typing import Any, Optional
from pandas import Series
from pandas_ta_classic.statistics.mad import mad
from pandas_ta_classic.utils import get_offset, verify_series


def md(
    close: Series,
    length: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Mean Deviation (MD)

    Rolling mean of absolute deviations from the rolling mean.
    Equivalent to ta.mad.  tulipy name: MD.
    """
    length = int(length) if length and length > 0 else 30
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None:
        return None

    result = mad(close, length=length, offset=offset)
    if result is None:
        return None

    result.name = f"MD_{length}"
    result.category = "statistics"
    return result


md.__doc__ = """Mean Deviation (MD)

Rolling mean of absolute deviations from the rolling mean.
Equivalent to ta.mad.  tulipy name: MD.

Args:
    close (pd.Series): Series of 'close' prices
    length (int): Lookback period. Default: 30
    offset (int): Periods to offset. Default: 0

Returns:
    pd.Series
"""
