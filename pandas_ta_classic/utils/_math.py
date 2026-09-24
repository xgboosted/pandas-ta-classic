import logging
from collections.abc import Callable
from math import comb, floor as mfloor
from sys import float_info as sflt
from typing import Any

import numpy as np
from pandas import DataFrame, Series

from ._core import _bool_param, _pos_int, verify_series

logger = logging.getLogger(__name__)


def np_rolling_moments(values: np.ndarray, length: int, *orders: int, min_periods: int | None = None) -> tuple[np.ndarray, ...]:
    """Rolling raw central-moment sums using pure numpy.

    Returns one float64 array per *order*, each of ``len(values)`` elements.
    Positions with fewer than ``min_periods`` (default: ``length``) valid
    observations are set to NaN.

    Each returned array contains **raw sums** of mean-centred deviations::

        result[i] = sum((window - mean(window)) ** k)

    These are *not* normalised statistical moments.  Callers such as
    ``kurtosis`` and ``skew`` apply the bias-correction factors themselves.

    Using numpy instead of ``pandas.rolling`` ensures cross-version
    determinism (pandas 2.x vs 3.x can round higher-order moments
    differently).
    """

    if min_periods is None:
        min_periods = length

    arr = values.astype(np.float64)
    n = len(arr)

    # Pre-allocate output arrays filled with NaN.
    results: list[np.ndarray] = [np.full(n, np.nan, dtype=np.float64) for _ in orders]

    # Vectorised computation over all full-length windows.
    if n >= length:
        windows = np.lib.stride_tricks.sliding_window_view(arr, length)
        mean = windows.mean(axis=1, keepdims=True)
        dev = windows - mean
        for i, k in enumerate(orders):
            results[i][length - 1 :] = (dev**k).sum(axis=1)

    # Scalar computation for partial windows when min_periods < length.
    if min_periods < length:
        for pos in range(min_periods - 1, min(length - 1, n)):
            window = arr[: pos + 1]
            dev = window - window.mean()
            for i, k in enumerate(orders):
                results[i][pos] = (dev**k).sum()

    return tuple(results)


def combination(*, n: int = 1, r: int = 0, repetition: bool = False, multichoose: bool = False) -> int:
    """nCr combinatorics — wraps math.comb. ``multichoose`` is an alias for ``repetition``."""
    n = _pos_int(n, 1, "n", gt=None, ge=0)
    r = _pos_int(r, 0, "r", gt=None, ge=0)
    repetition = _bool_param(repetition, False, "repetition")
    multichoose = _bool_param(multichoose, False, "multichoose")
    if repetition or multichoose:
        return comb(n + r - 1, r) if n + r > 0 else 1  # choosing 0 of 0 kinds: one way
    return comb(n, r)


def fibonacci(n: int = 2, *, zero: bool = False, weighted: bool = False) -> np.ndarray:
    """Fibonacci Sequence as a numpy array"""
    n = _pos_int(n, 2, "n", gt=None, ge=0)
    zero = _bool_param(zero, False, "zero")
    weighted = _bool_param(weighted, False, "weighted")

    if zero:
        a, b = 0, 1
    else:
        n -= 1
        a, b = 1, 1

    result = np.array([a])
    for _ in range(n):
        a, b = b, a + b
        result = np.append(result, a)

    if weighted:
        fib_sum: float = np.sum(result)
        if fib_sum > 0:
            return result / fib_sum
        return result
    return result


def linear_regression(x: Series, y: Series) -> dict:
    """Classic Linear Regression using Numpy"""
    x, y = verify_series(x), verify_series(y)
    m, n = x.size, y.size

    if m != n:
        raise ValueError(f"linear_regression() x and y must have equal length, got {m} and {n}")

    return _linear_regression_np(x, y)


def pascals_triangle(n: int | None = None, *, weighted: bool = False, inverse: bool = False) -> np.ndarray | None:
    """Pascal's Triangle

    Returns a numpy array of the nth row of Pascal's Triangle.
    n=4  => triangle: [1, 4, 6, 4, 1]
         => weighted: [0.0625, 0.25, 0.375, 0.25, 0.0625]
         => inverse weighted: [0.9375, 0.75, 0.625, 0.75, 0.9375]
    """
    n = _pos_int(n, 0, "n", gt=None, ge=0)
    weighted = _bool_param(weighted, False, "weighted")
    inverse = _bool_param(inverse, False, "inverse")

    # Calculation
    triangle = np.array([combination(n=n, r=i) for i in range(n + 1)])
    triangle_sum: float = np.sum(triangle)
    triangle_weights = triangle / triangle_sum
    inverse_weights = 1 - triangle_weights

    if inverse and not weighted:
        # it used to return None
        raise ValueError("pascals_triangle() inverse=True needs weighted=True")
    if weighted and inverse:
        return inverse_weights
    if weighted:
        return triangle_weights

    return triangle


def symmetric_triangle(n: int | None = None, *, weighted: bool = False) -> list[int] | np.ndarray | None:
    """Symmetric Triangle with n >= 2

    Returns a numpy array of the nth row of Symmetric Triangle.
    n=4  => triangle: [1, 2, 2, 1]
         => weighted: [0.16666667 0.33333333 0.33333333 0.16666667]
    """
    n = _pos_int(n, 2, "n")  # n=0 used to return None
    weighted = _bool_param(weighted, False, "weighted")

    triangle = None
    if n == 1:
        triangle = [1]

    if n == 2:
        triangle = [1, 1]

    if n > 2:
        if n % 2 == 0:
            front = [i + 1 for i in range(mfloor(n / 2))]
            triangle = front + front[::-1]
        else:
            front = [i + 1 for i in range(mfloor(0.5 * (n + 1)))]
            triangle = front.copy()
            front.pop()
            triangle += front[::-1]

    if weighted and isinstance(triangle, list):
        triangle_arr: np.ndarray = np.array(triangle)
        triangle_sum: float = float(np.sum(triangle_arr))
        triangle_weights: np.ndarray = triangle_arr / triangle_sum
        return triangle_weights

    return triangle


def weights(w: Any) -> Callable[[Any], Any]:
    """Calculates the dot product of weights with values x"""

    def _dot(x: Any) -> Any:
        return np.dot(w, x)

    return _dot


def zero(x: float) -> float:
    """If the value is close to zero, then return zero. Otherwise return itself."""
    return 0 if abs(x) < sflt.epsilon else x


def df_error_analysis(dfA: DataFrame, dfB: DataFrame, *, corr_method: str = "pearson", plot: bool = False, triangular: bool = False) -> DataFrame:
    """Correlation between two DataFrames, used by the test suite for oracle parity checks."""
    plot = _bool_param(plot, False, "plot")
    triangular = _bool_param(triangular, False, "triangular")

    # Find their differences and correlation
    diff = dfA - dfB
    corr = dfA.corr(dfB, method=corr_method)

    # For plotting
    if plot:
        diff.hist()
        if diff[diff > 0].any():
            diff.plot(kind="kde")

    if triangular:
        return corr.where(np.triu(np.ones(corr.shape)).astype(bool))

    return corr


def _linear_regression_np(x: Series, y: Series) -> dict:
    """Simple Linear Regression using Numpy for two 1d arrays."""
    result = {"a": np.nan, "b": np.nan, "r": np.nan, "t": np.nan, "line": np.nan}
    x_sum = x.sum()
    y_sum = y.sum()

    # A constant x (no variance) makes correlation and slope undefined; skip it.
    # The previous guard ``int(x_sum) != 0`` also skipped x whose values summed
    # to less than 1 in absolute value (daily benchmark returns), so the
    # regression never ran for Jensen's alpha.
    if x.std() != 0:
        # 1st row, 2nd col value corr(x, y)
        r = np.corrcoef(x, y)[0, 1]

        m = x.size
        r_mix = m * (x * y).sum() - x_sum * y_sum
        b = r_mix / (m * (x * x).sum() - x_sum * x_sum)
        a = y.mean() - b * x.mean()
        line = a + b * x

        _np_err = np.seterr()
        np.seterr(divide="ignore", invalid="ignore")
        result = {
            "a": a,
            "b": b,
            "r": r,
            "t": r / np.sqrt((1 - r * r) / (m - 2)),
            "line": line,
        }
        np.seterr(divide=_np_err["divide"], invalid=_np_err["invalid"])

    return result
