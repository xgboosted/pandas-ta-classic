from typing import Any

from pandas import Series

from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series


def mult(
    series_a: Series,
    series_b: Series,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Element-wise multiplication of two Series (TA-Lib: MULT)."""
    series_a = verify_series(series_a)
    series_b = verify_series(series_b)
    if series_a is None or series_b is None:
        return None
    offset = get_offset(offset)
    result = series_a * series_b
    result = apply_offset(result, offset)
    result = apply_fill(result, **kwargs)
    result.name = "MULT"
    result.category = "math"
    return result
