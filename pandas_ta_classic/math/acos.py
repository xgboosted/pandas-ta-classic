from typing import Any

import numpy as np
from pandas import Series

from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series


def acos(close: Series, offset: int | None = None, **kwargs: Any) -> Series | None:
    """Vector Trigonometric ACos (TA-Lib: ACOS)."""
    close = verify_series(close)
    if close is None:
        return None
    offset = get_offset(offset)
    result = close.apply(np.arccos)
    result = apply_offset(result, offset)
    result = apply_fill(result, **kwargs)
    result.name = "ACOS"
    result.category = "math"
    return result
