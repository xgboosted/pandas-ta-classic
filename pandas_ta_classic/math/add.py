from typing import Any

from pandas import Series

from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import nan_on_short_input


@nan_on_short_input
def add(
    series_a: Series,
    series_b: Series,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Element-wise addition of two Series (TA-Lib: ADD)."""
    series_a = verify_series(series_a)
    series_b = verify_series(series_b)
    if series_a is None or series_b is None:
        return None
    offset = get_offset(offset)
    result = series_a + series_b
    result = apply_offset(result, offset)
    result = apply_fill(result, **kwargs)
    result.name = "ADD"
    result.category = "math"
    return result


add.__doc__ = """Element-wise addition of two Series (TA-Lib: ADD)

Calculation:
    ADD = series_a + series_b

Args:
    series_a (pd.Series): The first addend
    series_b (pd.Series): The second addend
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
