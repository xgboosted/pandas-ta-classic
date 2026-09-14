from typing import Any

import numpy as np
from pandas import Series

from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _pos_int, _sliding_argextreme, nan_on_short_input


@nan_on_short_input
def maxindex(
    close: Series,
    length: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Window-relative index of the Maximum value over *length* periods.

    Named after TA-Lib's MAXINDEX, but the convention differs on purpose: this
    returns the 0-based index *within* the rolling window (0..length-1), i.e.
    how many bars back the high sits, not TA-Lib's absolute array index. There
    is deliberately no ``talib`` passthrough — TA-Lib's MAXINDEX would return
    different values, and neither tulipy nor Tulip Indicators expose an
    equivalent.
    """
    length = _pos_int(length, 30, "length")
    close = verify_series(close, length)
    offset = get_offset(offset)
    if close is None:
        return None
    result = _sliding_argextreme(close, length, np.argmax)
    result = apply_offset(result, offset)
    result = apply_fill(result, **kwargs)
    result.name = f"MAXINDEX_{length}"
    result.category = "math"
    return result
