# Mean Deviation (MD)
from typing import Any

from pandas import Series

from pandas_ta_classic.statistics.mad import mad
from pandas_ta_classic.utils import get_offset, verify_series
from pandas_ta_classic.utils._core import _pos_int, nan_on_short_input


@nan_on_short_input
def md(
    close: Series,
    length: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Mean Deviation (MD)

    Rolling mean of absolute deviations from the rolling mean.
    Equivalent to ta.mad.  tulipy name: MD.
    """
    length = _pos_int(length, 30, "length")
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None:
        return None

    # Only the fill reaches mad(): its min_periods option is not part of md's API.
    fill_kwargs = {key: kwargs[key] for key in ("fillna", "fill_method") if key in kwargs}
    result = mad(close, length=length, offset=offset, **fill_kwargs)
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

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series
"""
