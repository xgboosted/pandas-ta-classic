from typing import Any

from pandas import DataFrame, Series

from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _pos_int, nan_on_short_input


@nan_on_short_input
def minmax(
    close: Series,
    length: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> DataFrame | None:
    """Rolling Min and Max over *length* periods (TA-Lib: MINMAX).

    Returns a DataFrame with columns ``MIN_<n>`` and ``MAX_<n>``.
    """
    length = _pos_int(length, 30, "length")
    close = verify_series(close, length)
    offset = get_offset(offset)
    if close is None:
        return None
    mn = close.rolling(length).min()
    mx = close.rolling(length).max()
    mn, mx = apply_offset([mn, mx], offset)
    mn, mx = apply_fill([mn, mx], **kwargs)
    mn.name = f"MIN_{length}"
    mx.name = f"MAX_{length}"
    df = DataFrame({mn.name: mn, mx.name: mx})
    df.name = f"MINMAX_{length}"
    df.category = "math"
    return df
