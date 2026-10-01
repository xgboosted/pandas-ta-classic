# Linear Regression Intercept (LINEARREG_INTERCEPT)
from typing import Any

from pandas import Series

from pandas_ta_classic.overlap.linreg import linreg
from pandas_ta_classic.utils import get_offset, verify_series
from pandas_ta_classic.utils._core import _pos_int, nan_on_short_input


@nan_on_short_input
def linregintercept(
    close: Series,
    length: int | None = None,
    talib: bool | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Linear Regression Intercept (LINEARREG_INTERCEPT)

    The y-intercept of the linear regression line.
    TA-Lib name: LINEARREG_INTERCEPT.
    """
    length = _pos_int(length, 14, "length")
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None:
        return None

    # Only the fill reaches linreg(): its mode flags (slope, intercept, angle, ...)
    # would otherwise change which line this function returns.
    fill_kwargs = {key: kwargs[key] for key in ("fillna", "fill_method") if key in kwargs}
    return linreg(close, length=length, talib=talib, offset=offset, intercept=True, **fill_kwargs)


linregintercept.__doc__ = """Linear Regression Intercept (LINEARREG_INTERCEPT)

Returns the y-intercept of the linear regression line over the last
*length* bars.  Equivalent to ta.linreg(..., intercept=True).

Args:
    close (pd.Series): Series of 'close' prices
    length (int): Lookback period. Default: 14
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: False
    offset (int): Periods to offset. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series
"""
