# -*- coding: utf-8 -*-
# Standard Error (STDERR)
from typing import Any, Optional

from numpy import sqrt as npSqrt
from pandas import Series

from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series


def stderr(
    close: Series,
    length: Optional[int] = None,
    ddof: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Standard Error (STDERR)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 14
    ddof = int(ddof) if isinstance(ddof, int) and ddof >= 0 and ddof < length else 1
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result
    stderr_ = close.rolling(length).std(ddof=ddof) / npSqrt(length)

    # Offset
    stderr_ = apply_offset(stderr_, offset)

    stderr_ = apply_fill(stderr_, **kwargs)

    # Name and Categorize it
    stderr_.name = f"STDERR_{length}"
    stderr_.category = "statistics"

    return stderr_


stderr.__doc__ = """Standard Error (STDERR)

Standard Error is the standard deviation of the sample divided by the square
root of the sample size. It estimates the precision of the sample mean as an
estimate of the population mean.

STDERR = StdDev(close, length) / sqrt(length)

Sources:
    https://en.wikipedia.org/wiki/Standard_error

Args:
    close (pd.Series): Price series.
    length (int): Rolling window period. Default: 14
    ddof (int): Degrees of freedom for std. Default: 1
    offset (int): Result offset. Default: 0

Returns:
    pd.Series: STDERR values.
"""
