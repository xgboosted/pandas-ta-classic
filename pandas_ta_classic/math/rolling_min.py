from typing import Any, Optional

from pandas import Series

from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series


def rolling_min(
    close: Series,
    length: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Rolling Minimum over *length* periods (TA-Lib: MIN)."""
    length = int(length) if length and length > 0 else 30
    close = verify_series(close, length)
    offset = get_offset(offset)
    if close is None:
        return None
    result = close.rolling(length).min()
    result = apply_offset(result, offset)
    result = apply_fill(result, **kwargs)
    result.name = f"MIN_{length}"
    result.category = "math"
    return result
