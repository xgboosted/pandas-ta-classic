from typing import Any

import numpy as np
from pandas import Series

from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _pos_int, _sliding_argextreme, nan_on_short_input, skip_leading_nan


@nan_on_short_input
@skip_leading_nan("close")
def maxindex(
    close: Series,
    length: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Window-relative index of the Maximum value over *length* periods.

    Named after TA-Lib's MAXINDEX, but the convention differs on purpose: this
    returns the 0-based position of the maximum *within* the rolling window,
    counted from its oldest bar: 0 is the bar ``length - 1`` bars back, and
    ``length - 1`` is the current bar. On a tie the oldest bar wins, and a
    window holding a NaN gives NaN. TA-Lib returns the absolute array index
    instead, so at bar ``i`` (its position in the series)
    ``talib = result + i - length + 1``, except on a tie, where TA-Lib may pick
    a later bar. There is deliberately no ``talib`` passthrough, and neither
    tulipy nor Tulip Indicators expose an equivalent.
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
