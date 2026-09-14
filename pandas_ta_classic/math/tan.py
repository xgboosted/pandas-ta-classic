from typing import Any

import numpy as np
from pandas import Series

from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import nan_on_short_input


@nan_on_short_input
def tan(close: Series, offset: int | None = None, **kwargs: Any) -> Series | None:
    """Vector Trigonometric Tan (TA-Lib: TAN)."""
    close = verify_series(close)
    if close is None:
        return None
    offset = get_offset(offset)
    result = close.apply(np.tan)
    result = apply_offset(result, offset)
    result = apply_fill(result, **kwargs)
    result.name = "TAN"
    result.category = "math"
    return result
