from typing import Any, Optional

import numpy as np
from pandas import Series

from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series


def todeg(close: Series, offset: Optional[int] = None, **kwargs: Any) -> Optional[Series]:
    """Vector Degrees conversion (tulipy: TODEG). Converts radians to degrees."""
    close = verify_series(close)
    if close is None:
        return None
    offset = get_offset(offset)
    result = close.apply(np.degrees)
    result = apply_offset(result, offset)
    result = apply_fill(result, **kwargs)
    result.name = "TODEG"
    result.category = "math"
    return result
