from typing import Any, Optional

from pandas import Series

from pandas_ta_classic import Imports
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series


def rolling_min(
    close: Series,
    length: Optional[int] = None,
    talib: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Rolling Minimum over *length* periods (TA-Lib: MIN)."""
    length = int(length) if length is not None else 30
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    close = verify_series(close, length)
    offset = get_offset(offset)
    mode_talib = bool(talib) if isinstance(talib, bool) else False
    if close is None:
        return None
    if Imports["talib"] and mode_talib:
        from talib import MIN

        result = MIN(close, length)
    else:
        result = close.rolling(length).min()
    result = apply_offset(result, offset)
    result = apply_fill(result, **kwargs)
    result.name = f"MIN_{length}"
    result.category = "math"
    return result
