# Skew (SKEW)
from typing import Any

import numpy as np
from pandas import Series

from pandas_ta_classic.utils import (
    apply_fill,
    apply_offset,
    degenerate_zero,
    get_offset,
    np_rolling_moments,
    verify_series,
)
from pandas_ta_classic.utils._core import _pos_int, nan_on_short_input


@nan_on_short_input
def skew(
    close: Series,
    length: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Skew"""
    # Validate Arguments
    # The adjusted Fisher-Pearson skew divides by (n - 2), so it is undefined
    # below three points; length == 2 used to return a column of +-inf.
    length = _pos_int(length, 30, "length", gt=2)
    # min_periods below 1 read the loop start as -1 and made the last bar NaN;
    # above length it was silently ignored. Both are caller errors.
    min_periods = _pos_int(kwargs.get("min_periods"), length, "min_periods", gt=None, ge=1, lt=length + 1)
    close = verify_series(close, max(length, min_periods))
    offset = get_offset(offset)

    if close is None:
        return None

    # Pure numpy rolling skewness (adjusted Fisher-Pearson) for cross-version
    # determinism.
    m2, m3 = np_rolling_moments(close.values, length, 2, 3, min_periods=min_periods)
    # n_eff[i] is the actual window size at position i (scalar for the common
    # case where min_periods == length).
    n_eff: np.ndarray | np.float64
    if min_periods < length:
        n_eff = np.full(len(close), np.float64(length))
        for pos in range(min_periods - 1, min(length - 1, len(close))):
            n_eff[pos] = pos + 1
    else:
        n_eff = np.float64(length)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = n_eff * np.sqrt(n_eff - 1) / (n_eff - 2) * m3 / m2**1.5
    # A window with zero variance has m2 == 0, so the division is 0/0. It reads
    # 0.0, the convention TA-Lib applies throughout for a degenerate window;
    # see tests/test_degenerate_input.py. degenerate_zero catches the ~1e-33
    # residue a flat 0.3 leaves in m2.
    result = np.where(degenerate_zero(m2), 0.0, result)
    if min_periods < length:
        # Only then can a window be narrower than the three points the formula
        # needs, and those leading bars have no skew to report. Guarded, so the
        # default path keeps the scalar `n_eff` free of an array copy.
        result = np.where(n_eff < 3, np.nan, result)
    skew = Series(result, index=close.index, dtype=np.float64)

    # Offset
    skew = apply_offset(skew, offset)

    skew = apply_fill(skew, **kwargs)

    # Name & Category
    skew.name = f"SKEW_{length}"
    skew.category = "statistics"

    return skew


skew.__doc__ = """Rolling Skew

Sources:

Calculation:
    Default Inputs:
        length=30
    SKEW = close.rolling(length).skew()

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Must be > 2, the narrowest window a skew is
        defined on. Default: 30
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
