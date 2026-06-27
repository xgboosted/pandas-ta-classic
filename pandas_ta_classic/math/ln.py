from typing import Any, Optional

import numpy as np
from pandas import Series

from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series


def ln(close: Series, offset: Optional[int] = None, **kwargs: Any) -> Optional[Series]:
    """Vector Log Natural (TA-Lib: LN)."""
    close = verify_series(close)
    if close is None:
        return None
    offset = get_offset(offset)
    result = close.apply(np.log)
    result = apply_offset(result, offset)
    result = apply_fill(result, **kwargs)
    result.name = "LN"
    result.category = "math"
    return result
