from typing import Any

import numpy as np
from pandas import Series

from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series


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
