from typing import Any, Optional

import numpy as np
from pandas import Series

from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _sliding_argextreme


def minindex(
    close: Series,
    length: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Index of the Minimum value over *length* periods (TA-Lib: MININDEX).

    Returns 0-based index within the rolling window.
    """
    length = int(length) if length and length > 0 else 30
    close = verify_series(close, length)
    offset = get_offset(offset)
    if close is None:
        return None
    result = _sliding_argextreme(close, length, np.argmin)
    result = apply_offset(result, offset)
    result = apply_fill(result, **kwargs)
    result.name = f"MININDEX_{length}"
    result.category = "math"
    return result
