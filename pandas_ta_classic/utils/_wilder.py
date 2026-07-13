"""Wilder's cumulative smoothing — TA-Lib-exact implementation.

Unlike EMA-seeded RMA (``overlap/rma.py``), this uses the Wilder convention:

    seed = sum(raw[1 : length])          # skip index 0
    smoothed[t] = smoothed[t-1] * (1 - 1/length) + raw[t]

This is the algorithm used internally by TA-Lib's PLUS_DM, MINUS_DM,
DX, ADX, ADXR (Wilder-variant RSI), ATR, and NATR.

References
----------
* Wilder, J. Welles. *New Concepts in Technical Trading Systems* (1978)
"""

import numpy as np
from pandas import Series

from pandas_ta_classic.utils._njit import njit


@njit(cache=True)
def _wilder_smooth_nb(arr: np.ndarray, length: int, seed: float) -> np.ndarray:
    # ``seed`` is pre-computed by the caller with numpy so the cumulative sum
    # matches bit-for-bit; the kernel only runs the deterministic scalar
    # recursion.
    n = len(arr)
    result = np.full(n, np.nan)
    result[length - 1] = seed

    value = seed
    for i in range(length, n):
        raw_i = arr[i]
        if np.isnan(raw_i):
            result[i] = value  # carry forward on NaN input
        else:
            value = value - value / length + raw_i
            result[i] = value

    return result


def wilder_smooth(raw: Series, length: int) -> Series:
    """Apply Wilder's cumulative smoothing to *raw*.

    Parameters
    ----------
    raw : pd.Series
        Pre-processed series (directional movement, true range, etc.).
        May contain NaN; the seed is computed from positions ``[1:length]``.
    length : int
        Smoothing period (e.g. 14 for ADX, 5 for Fast Stochastic).

    Returns
    -------
    pd.Series
        Wilder-smoothed series.  The first ``length`` bars are NaN;
        bar ``length - 1`` holds the seed value.  The output has the
        same index as *raw*.
    """
    arr = raw.to_numpy(dtype=float)
    n = len(arr)
    if n < length:
        return Series(np.full(n, np.nan), index=raw.index)

    seed = float(np.nansum(arr[1:length]))
    result = _wilder_smooth_nb(arr, length, seed)
    return Series(result, index=raw.index)
