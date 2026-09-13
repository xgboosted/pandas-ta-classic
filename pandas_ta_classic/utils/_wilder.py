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
def _wilder_smooth_nb(arr: np.ndarray, length: int, seed: float, start: int) -> np.ndarray:
    # ``seed`` is pre-computed by the caller with numpy so the cumulative sum
    # matches bit-for-bit; the kernel only runs the deterministic scalar
    # recursion. ``start`` is the first raw position summed into the seed.
    n = len(arr)
    result = np.full(n, np.nan)
    result[start + length - 2] = seed

    value = seed
    for i in range(start + length - 1, n):
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
        Index 0 is always skipped (it is undefined for a diff-based series),
        and so is any longer leading NaN run: the seed is the sum of the
        ``length - 1`` values that follow it, as TA-Lib does for chained input.
    length : int
        Smoothing period (e.g. 14 for ADX, 5 for Fast Stochastic).

    Returns
    -------
    pd.Series
        Wilder-smoothed series.  Bars before the seed are NaN; on input
        with a single undefined first bar the seed sits at ``length - 1``.
        The output has the same index as *raw*.
    """
    arr = raw.to_numpy(dtype=float)
    n = len(arr)
    finite = np.flatnonzero(np.isfinite(arr))
    start = max(int(finite[0]), 1) if finite.size else n
    if start + length - 1 > n:
        return Series(np.full(n, np.nan), index=raw.index)

    seed = float(np.nansum(arr[start : start + length - 1]))
    result = _wilder_smooth_nb(arr, length, seed, start)
    return Series(result, index=raw.index)


def wilder_di(pos: Series, neg: Series, tr: Series, length: int, scalar: float) -> tuple[Series, Series]:
    """+DI and -DI from raw directional movement and true range (TA-Lib-exact).

    The three series are Wilder-smoothed with :func:`wilder_smooth`; the seed
    bar itself is not reported, because TA-Lib's PLUS_DI/MINUS_DI lookback is
    ``length``, one bar after the smoothed seed.
    """
    tr_s = wilder_smooth(tr, length)
    dmp = scalar * wilder_smooth(pos, length) / tr_s
    dmn = scalar * wilder_smooth(neg, length) / tr_s
    seed = tr_s.first_valid_index()
    if seed is not None:
        dmp[seed] = np.nan
        dmn[seed] = np.nan
    return dmp, dmn
