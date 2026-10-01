# Kurtosis (KURTOSIS)
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
def kurtosis(
    close: Series,
    length: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Kurtosis"""
    # Validate Arguments
    # Excess kurtosis divides by (n - 2)(n - 3), so it is undefined below four
    # points; length <= 3 used to return an all-NaN column.
    length = _pos_int(length, 30, "length", gt=3)
    # min_periods below 1 read the loop start as -1 and made the last bar NaN;
    # above length it was silently ignored. Both are caller errors.
    min_periods = _pos_int(kwargs.get("min_periods"), length, "min_periods", gt=None, ge=1, lt=length + 1)
    close = verify_series(close, max(length, min_periods))
    offset = get_offset(offset)

    if close is None:
        return None

    # Pure numpy rolling excess kurtosis (Fisher) for cross-version determinism.
    m2, m4 = np_rolling_moments(close.values, length, 2, 4, min_periods=min_periods)
    # n_eff[i] is the actual window size used at position i.  When
    # min_periods == length (the default) every position uses length, so a
    # scalar is sufficient and avoids the array-allocation overhead.
    n_eff: np.ndarray | np.float64
    if min_periods < length:
        n_eff = np.full(len(close), np.float64(length))
        for pos in range(min_periods - 1, min(length - 1, len(close))):
            n_eff[pos] = pos + 1
    else:
        n_eff = np.float64(length)
    with np.errstate(divide="ignore", invalid="ignore"):
        numer = n_eff * (n_eff + 1) * (n_eff - 1) * m4
        denom = (n_eff - 2) * (n_eff - 3) * m2**2
        adj = 3.0 * (n_eff - 1) ** 2 / ((n_eff - 2) * (n_eff - 3))
        result = numer / denom - adj
    # A window with zero variance has m2 == 0, so the division is 0/0. It reads
    # 0.0, the convention TA-Lib applies throughout for a degenerate window;
    # see tests/test_degenerate_input.py. degenerate_zero catches the ~1e-33
    # residue a flat 0.3 leaves in m2. Masking on m2 rather than denom avoids
    # reading 0.0 for every window narrower than four bars as well, where the
    # formula is undefined rather than degenerate.
    result = np.where(degenerate_zero(m2), 0.0, result)
    if min_periods < length:
        # Only then can a window be narrower than the four points the formula
        # needs, and those leading bars have no kurtosis to report. Guarded, so
        # the default path keeps the scalar `n_eff` free of an array copy.
        result = np.where(n_eff < 4, np.nan, result)
    kurtosis = Series(result, index=close.index, dtype=np.float64)

    # Offset
    kurtosis = apply_offset(kurtosis, offset)

    kurtosis = apply_fill(kurtosis, **kwargs)

    # Name & Category
    kurtosis.name = f"KURT_{length}"
    kurtosis.category = "statistics"

    return kurtosis


kurtosis.__doc__ = """Rolling Kurtosis

Sources:

Calculation:
    Default Inputs:
        length=30
    KURTOSIS = close.rolling(length).kurt()

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Must be > 3, the narrowest window an excess
        kurtosis is defined on. Default: 30
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
