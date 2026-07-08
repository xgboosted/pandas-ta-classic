import logging
from math import comb
from math import floor as mfloor
from sys import float_info as sflt
from typing import Any, Callable, Optional, Union

import numpy as np
from pandas import DataFrame, Series

from ._core import verify_series

logger = logging.getLogger(__name__)


def np_rolling_moments(values: np.ndarray, length: int, *orders: int, min_periods: Optional[int] = None) -> tuple[np.ndarray, ...]:
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
    from numpy.lib.stride_tricks import sliding_window_view

    if min_periods is None:
        min_periods = length

    arr = values.astype(np.float64)
    n = len(arr)

    # Pre-allocate output arrays filled with NaN.
    results: list[np.ndarray] = [np.full(n, np.nan, dtype=np.float64) for _ in orders]

    # Vectorised computation over all full-length windows.
    if n >= length:
        windows = sliding_window_view(arr, length)
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


def combination(**kwargs: Any) -> int:
    """nCr combinatorics — wraps math.comb."""
    n = int(abs(kwargs.pop("n", 1)))
    r = int(abs(kwargs.pop("r", 0)))
    if kwargs.pop("repetition", False):
        n = n + r - 1
    return comb(n, r)


def fibonacci(n: int = 2, **kwargs: Any) -> np.ndarray:
    """Fibonacci Sequence as a numpy array"""
    n = int(n) if n >= 0 else 2

    zero = kwargs.pop("zero", False)
    if zero:
        a, b = 0, 1
    else:
        n -= 1
        a, b = 1, 1

    result = np.array([a])
    for _ in range(0, n):
        a, b = b, a + b
        result = np.append(result, a)

    weighted = kwargs.pop("weighted", False)
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
        logger.error("Linear Regression X and y have unequal total observations: %d != %d", m, n)
        return {}

    return _linear_regression_np(x, y)


def pascals_triangle(n: Optional[int] = None, **kwargs: Any) -> Optional[np.ndarray]:
    """Pascal's Triangle

    Returns a numpy array of the nth row of Pascal's Triangle.
    n=4  => triangle: [1, 4, 6, 4, 1]
         => weighted: [0.0625, 0.25, 0.375, 0.25, 0.0625]
         => inverse weighted: [0.9375, 0.75, 0.625, 0.75, 0.9375]
    """
    n = int(abs(n)) if n is not None else 0

    # Calculation
    triangle = np.array([combination(n=n, r=i) for i in range(0, n + 1)])
    triangle_sum: float = np.sum(triangle)
    triangle_weights = triangle / triangle_sum
    inverse_weights = 1 - triangle_weights

    weighted = kwargs.pop("weighted", False)
    inverse = kwargs.pop("inverse", False)
    if weighted and inverse:
        return inverse_weights
    if weighted:
        return triangle_weights
    if inverse:
        return None

    return triangle


def symmetric_triangle(n: Optional[int] = None, **kwargs: Any) -> Optional[Union[list[int], np.ndarray]]:
    """Symmetric Triangle with n >= 2

    Returns a numpy array of the nth row of Symmetric Triangle.
    n=4  => triangle: [1, 2, 2, 1]
         => weighted: [0.16666667 0.33333333 0.33333333 0.16666667]
    """
    n = int(abs(n)) if n is not None else 2

    triangle = None
    if n == 2:
        triangle = [1, 1]

    if n > 2:
        if n % 2 == 0:
            front = [i + 1 for i in range(0, mfloor(n / 2))]
            triangle = front + front[::-1]
        else:
            front = [i + 1 for i in range(0, mfloor(0.5 * (n + 1)))]
            triangle = front.copy()
            front.pop()
            triangle += front[::-1]

    if kwargs.pop("weighted", False) and isinstance(triangle, list):
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


def zero(x: Union[int, float]) -> Union[int, float]:
    """If the value is close to zero, then return zero. Otherwise return itself."""
    return 0 if abs(x) < sflt.epsilon else x


def df_error_analysis(dfA: DataFrame, dfB: DataFrame, **kwargs: Any) -> DataFrame:
    """Correlation between two DataFrames, used by the test suite for oracle parity checks."""
    corr_method = kwargs.pop("corr_method", "pearson")

    # Find their differences and correlation
    diff = dfA - dfB
    corr = dfA.corr(dfB, method=corr_method)

    # For plotting
    if kwargs.pop("plot", False):
        diff.hist()
        if diff[diff > 0].any():
            diff.plot(kind="kde")

    if kwargs.pop("triangular", False):
        return corr.where(np.triu(np.ones(corr.shape)).astype(bool))

    return corr


def _linear_regression_np(x: Series, y: Series) -> dict:
    """Simple Linear Regression using Numpy for two 1d arrays."""
    result = {"a": np.nan, "b": np.nan, "r": np.nan, "t": np.nan, "line": np.nan}
    x_sum = x.sum()
    y_sum = y.sum()

    if int(x_sum) != 0:
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
