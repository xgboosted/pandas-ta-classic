from typing import Any, Optional

import numpy as np
from pandas import Series

from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series


def sin(close: Series, offset: Optional[int] = None, **kwargs: Any) -> Optional[Series]:
    """Vector Trigonometric Sin (TA-Lib: SIN)."""
    close = verify_series(close)
    if close is None:
        return None
    offset = get_offset(offset)
    result = close.apply(np.sin)
    result = apply_offset(result, offset)
    result = apply_fill(result, **kwargs)
    result.name = "SIN"
    result.category = "math"
    return result
