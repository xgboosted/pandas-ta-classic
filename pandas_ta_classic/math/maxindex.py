from typing import Any, Optional

import numpy as np
from pandas import Series

from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series


def maxindex(
    close: Series,
    length: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Window-relative index of the Maximum value over *length* periods.

    Named after TA-Lib's MAXINDEX, but the convention differs on purpose: this
    returns the 0-based index *within* the rolling window (0..length-1), i.e.
    how many bars back the high sits, not TA-Lib's absolute array index. There
    is deliberately no ``talib`` passthrough — TA-Lib's MAXINDEX would return
    different values, and neither tulipy nor Tulip Indicators expose an
    equivalent.
    """
    length = int(length) if length and length > 0 else 30
    close = verify_series(close, length)
    offset = get_offset(offset)
    if close is None:
        return None
    result = close.rolling(length).apply(np.argmax, raw=True)
    result = apply_offset(result, offset)
    result = apply_fill(result, **kwargs)
    result.name = f"MAXINDEX_{length}"
    result.category = "math"
    return result
