# Linear Regression Angle (LINEARREG_ANGLE)
from typing import Any

from pandas import Series

from pandas_ta_classic.overlap.linreg import linreg
from pandas_ta_classic.utils import get_offset, verify_series
from pandas_ta_classic.utils._core import _pos_int, nan_on_short_input


@nan_on_short_input
def linregangle(
    close: Series,
    length: int | None = None,
    talib: bool | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Linear Regression Angle (LINEARREG_ANGLE)

    The angle (in degrees) of the linear regression slope.
    TA-Lib name: LINEARREG_ANGLE.
    """
    length = _pos_int(length, 14, "length")
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None:
        return None

    return linreg(close, length=length, talib=talib, offset=offset, angle=True, degrees=True)


linregangle.__doc__ = """Linear Regression Angle (LINEARREG_ANGLE)

Returns the angle (in degrees) of the linear regression line over the
last *length* bars.  Equivalent to ta.linreg(..., angle=True).

Args:
    close (pd.Series): Series of 'close' prices
    length (int): Lookback period. Default: 14
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: False
    offset (int): Periods to offset. Default: 0

Returns:
    pd.Series
"""
