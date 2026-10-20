from typing import Any

import numpy as np
from pandas import Series

from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import nan_on_short_input


@nan_on_short_input
def npabs(close: Series, offset: int | None = None, **kwargs: Any) -> Series | None:
    """Vector Absolute Value (tulipy: ABS)."""
    close = verify_series(close)
    if close is None:
        return None
    offset = get_offset(offset)
    result = close.apply(np.abs)
    result = apply_offset(result, offset)
    result = apply_fill(result, **kwargs)
    result.name = "ABS"
    result.category = "math"
    return result


npabs.__doc__ = """Vector Absolute Value (tulipy: ABS)

Calculation:
    ABS = |close|

Args:
    close (pd.Series): Series of input values
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
