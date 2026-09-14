from typing import Any

from pandas import Series

from pandas_ta_classic import Imports
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _bool_param, _pos_int, nan_on_short_input


@nan_on_short_input
def rolling_min(
    close: Series,
    length: int | None = None,
    talib: bool | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Rolling Minimum over *length* periods (TA-Lib: MIN)."""
    length = _pos_int(length, 30, "length")
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    close = verify_series(close, length)
    offset = get_offset(offset)
    mode_talib = _bool_param(talib, False, "talib")
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
