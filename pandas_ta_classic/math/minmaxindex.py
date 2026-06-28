from typing import Any, Optional

import numpy as np
from pandas import DataFrame, Series

from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series


def minmaxindex(
    close: Series,
    length: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Rolling Min and Max indices over *length* periods (TA-Lib: MINMAXINDEX).

    Returns a DataFrame with columns ``MINIDX_<n>`` and ``MAXIDX_<n>``.
    """
    length = int(length) if length and length > 0 else 30
    close = verify_series(close, length)
    offset = get_offset(offset)
    if close is None:
        return None
    mn_idx = close.rolling(length).apply(np.argmin, raw=True)
    mx_idx = close.rolling(length).apply(np.argmax, raw=True)
    mn_idx, mx_idx = apply_offset([mn_idx, mx_idx], offset)
    mn_idx, mx_idx = apply_fill([mn_idx, mx_idx], **kwargs)
    mn_idx.name = f"MINIDX_{length}"
    mx_idx.name = f"MAXIDX_{length}"
    df = DataFrame({mn_idx.name: mn_idx, mx_idx.name: mx_idx})
    df.name = f"MINMAXINDEX_{length}"
    df.category = "math"
    return df
