from typing import Any, Optional

from pandas import Series

from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series


def rolling_max(
    close: Series,
    length: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Rolling Maximum over *length* periods (TA-Lib: MAX)."""
    length = int(length) if length is not None else 30
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    close = verify_series(close, length)
    offset = get_offset(offset)
    if close is None:
        return None
    result = close.rolling(length).max()
    result = apply_offset(result, offset)
    result = apply_fill(result, **kwargs)
    result.name = f"MAX_{length}"
    result.category = "math"
    return result
