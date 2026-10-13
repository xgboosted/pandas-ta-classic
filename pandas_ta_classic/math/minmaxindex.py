from typing import Any

import numpy as np
from pandas import DataFrame, Series

from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _pos_int, _sliding_argextreme, nan_on_short_input, skip_leading_nan


@nan_on_short_input
@skip_leading_nan("close")
def minmaxindex(
    close: Series,
    length: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> DataFrame | None:
    """Window-relative Min and Max indices over *length* periods.

    Returns a DataFrame with columns ``MINIDX_<n>`` and ``MAXIDX_<n>``, the
    values ``minindex`` and ``maxindex`` return: the 0-based position of the
    minimum and maximum *within* the rolling window, counted from its oldest
    bar (0 is the bar ``length - 1`` bars back, ``length - 1`` the current
    bar). On a tie the oldest bar wins, and a window holding a NaN gives NaN.

    Named after TA-Lib's MINMAXINDEX, but the convention differs on purpose:
    TA-Lib returns the absolute array index, so at bar ``i`` (its position in
    the series)
    ``talib = result + i - length + 1``, except on a tie, where TA-Lib may pick
    a later bar. There is deliberately no ``talib`` passthrough, and neither
    tulipy nor Tulip Indicators expose an equivalent.
    """
    length = _pos_int(length, 30, "length")
    close = verify_series(close, length)
    offset = get_offset(offset)
    if close is None:
        return None
    mn_idx = _sliding_argextreme(close, length, np.argmin)
    mx_idx = _sliding_argextreme(close, length, np.argmax)
    mn_idx, mx_idx = apply_offset([mn_idx, mx_idx], offset)
    mn_idx, mx_idx = apply_fill([mn_idx, mx_idx], **kwargs)
    mn_idx.name = f"MINIDX_{length}"
    mx_idx.name = f"MAXIDX_{length}"
    df = DataFrame({mn_idx.name: mn_idx, mx_idx.name: mx_idx})
    df.name = f"MINMAXINDEX_{length}"
    df.category = "math"
    return df
