# Slope (SLOPE)
from typing import Any

import numpy as np
from pandas import Series

from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _bool_param, _pos_int


def slope(
    close: Series,
    length: int | None = None,
    as_angle: bool | None = None,
    to_degrees: bool | None = None,
    vertical: bool | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Slope"""
    # Validate arguments
    length = _pos_int(length, 1, "length")
    # was bool(isinstance(x, bool)), which turned as_angle=False into True
    as_angle = _bool_param(as_angle, False, "as_angle")
    to_degrees = _bool_param(to_degrees, False, "to_degrees")
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result
    slope = close.diff(length) / length
    if as_angle:
        slope = slope.apply(np.arctan)
        if to_degrees:
            slope *= 180 / np.pi

    # Offset
    slope = apply_offset(slope, offset)

    slope = apply_fill(slope, **kwargs)

    # Name and Categorize it
    slope.name = f"SLOPE_{length}" if not as_angle else f"ANGLE{'d' if to_degrees else 'r'}_{length}"
    slope.category = "momentum"

    return slope


slope.__doc__ = """Slope

Returns the slope of a series of length n. Can convert the slope to angle.
Default: slope.

Sources: Algebra I

Calculation:
    Default Inputs:
        length=1
    slope = close.diff(length) / length

    if as_angle:
        slope = slope.apply(atan)
        if to_degrees:
            slope *= 180 / PI

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period.  Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    as_angle (value, optional): Converts slope to an angle. Default: False
    to_degrees (value, optional): Converts slope angle to degrees. Default: False
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
