from typing import Any

import numpy as np
from pandas import Series

from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import nan_on_short_input


@nan_on_short_input
def log10(close: Series, offset: int | None = None, **kwargs: Any) -> Series | None:
    """Vector Log10 (TA-Lib: LOG10)."""
    close = verify_series(close)
    if close is None:
        return None
    offset = get_offset(offset)
    result = close.apply(np.log10)
    result = apply_offset(result, offset)
    result = apply_fill(result, **kwargs)
    result.name = "LOG10"
    result.category = "math"
    return result


log10.__doc__ = """Vector Log10 (TA-Lib: LOG10)

Calculation:
    LOG10 = log10(close)

Args:
    close (pd.Series): Series of input values
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
