# Linear Regression Angle (LINEARREG_ANGLE)
from typing import Annotated, Any

from pandas import Series

from pandas_ta_classic.overlap.linreg import linreg
from pandas_ta_classic.utils import get_offset, verify_series
from pandas_ta_classic.utils._core import Bound, nan_on_short_input


@nan_on_short_input
def linregangle(
    close: Series,
    length: Annotated[int, Bound(gt=1)] = 14,
    talib: bool | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Linear Regression Angle (LINEARREG_ANGLE)

    The angle (in degrees) of the linear regression slope.
    TA-Lib name: LINEARREG_ANGLE.
    """
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None:
        return None

    # Only the fill reaches linreg(): its mode flags (slope, intercept, angle, ...)
    # would otherwise change which line this function returns.
    fill_kwargs = {key: kwargs[key] for key in ("fillna", "fill_method") if key in kwargs}
    return linreg(close, length=length, talib=talib, offset=offset, angle=True, degrees=True, **fill_kwargs)


linregangle.__doc__ = """Linear Regression Angle (LINEARREG_ANGLE)

Returns the angle (in degrees) of the linear regression line over the
last *length* bars.  Equivalent to ta.linreg(..., angle=True).

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
