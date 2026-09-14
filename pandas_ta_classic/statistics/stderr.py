# Standard Error (STDERR)
from typing import Any

import numpy as np
from pandas import Series

from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _pos_int, nan_on_short_input


@nan_on_short_input
def stderr(
    close: Series,
    length: int | None = None,
    ddof: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Standard Error (STDERR)"""
    # Validate Arguments
    length = _pos_int(length, 14, "length")
    ddof = _pos_int(ddof, 1, "ddof", gt=None, ge=0, lt=length)
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result
    stderr_ = close.rolling(length).std(ddof=ddof) / np.sqrt(length)

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

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: STDERR values.
"""
