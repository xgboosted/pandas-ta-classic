from typing import Any, Optional

import numpy as np
from pandas import Series

from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series


def cos(close: Series, offset: Optional[int] = None, **kwargs: Any) -> Optional[Series]:
    """Vector Trigonometric Cos (TA-Lib: COS)."""
    close = verify_series(close)
    if close is None:
        return None
    offset = get_offset(offset)
    result = close.apply(np.cos)
    result = apply_offset(result, offset)
    result = apply_fill(result, **kwargs)
    result.name = "COS"
    result.category = "math"
    return result
