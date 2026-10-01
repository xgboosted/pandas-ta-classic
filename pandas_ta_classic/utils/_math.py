import logging
from collections.abc import Callable
from math import comb, floor as mfloor
from sys import float_info as sflt
from typing import Any

import numpy as np
from pandas import DataFrame, Series

from ._core import _bool_param, _pos_int, _require_input, degenerate_zero, verify_series

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
    _require_input(values, "np_rolling_moments", "values")
    _require_input(length, "np_rolling_moments", "length")

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


def combination(*, n: int = 1, r: int = 0, repetition: bool = False) -> int:
    """nCr combinatorics — wraps math.comb.

    Note: the ``multichoose`` alias of ``repetition`` was removed in 0.9.0;
    passing it raises TypeError.
    """
    n = _pos_int(n, 1, "n", gt=None, ge=0)
    r = _pos_int(r, 0, "r", gt=None, ge=0)
    repetition = _bool_param(repetition, False, "repetition")
    if repetition:
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

    # Build in a Python list and convert once. The previous np.append() in the
    # loop was quadratic and, worse, kept the seed array's int64 dtype: from
    # n=91 the terms wrapped silently (fibonacci(91, weighted=True) summed to
    # -6.2e18, so fwma(length=91) returned nonsense) until the object-dtype
    # promotion at n>=93 happened to rescue it. Python ints do not overflow.
    terms = [a]
    for _ in range(n):
        a, b = b, a + b
        terms.append(a)

    if weighted:
        fib_sum = sum(terms)  # exact: a float64 sum loses precision past 2**53
        if fib_sum > 0:
            return np.array([term / fib_sum for term in terms], dtype=float)
        # A zero sum is reachable only for fibonacci(0, zero=True) ([0]); there
        # is no weight to distribute, so the weighted result is all zeros rather
        # than silently the unweighted int array (rule 1 forbids that fallback).
        return np.zeros(len(terms), dtype=float)

    return np.array(terms, dtype=object if terms[-1] > np.iinfo(np.int64).max else np.int64)


def linear_regression(x: Series, y: Series) -> dict:
    """Classic Linear Regression using Numpy

    Args:
        x (pd.Series): Independent values
        y (pd.Series): Dependent values, on the same index as ``x``

    Returns:
        dict: ``a`` (intercept), ``b`` (slope), ``r`` (correlation), ``t``
        (t-statistic of ``r``) and ``line`` (the fitted values). A flat ``x``
        (constant up to rounding at its own scale) returns NaN for all five; a
        flat ``y`` gives ``r`` and ``t`` NaN.

    Raises:
        ValueError: ``x`` and ``y`` differ in length or index, have fewer than
            3 points, or contain NaN or an infinite value.
    """
    _require_input(x, "linear_regression", "x")
    _require_input(y, "linear_regression", "y")
    x, y = verify_series(x), verify_series(y)
    m, n = x.size, y.size

    if m != n:
        raise ValueError(f"linear_regression() x and y must have equal length, got {m} and {n}")
    # x * y aligns on the index, so two different indexes summed an empty product
    # into a wrong slope while np.corrcoef, which is positional, still gave r.
    if not x.index.equals(y.index):
        raise ValueError("linear_regression() x and y must share the same index")
    # t has m - 2 degrees of freedom: two points gave t = 0.0, fewer gave NaN.
    if m < 3:
        raise ValueError(f"linear_regression() needs at least 3 points, got {m}")
    # The sums skip NaN while m counts it, so a NaN gave a wrong slope and intercept.
    # An infinite value turned all five results into NaN, with numpy warnings.
    for label, series in (("x", x), ("y", y)):
        gaps = int(series.isna().sum())
        if gaps:
            raise ValueError(f"linear_regression() {label} has {gaps} missing value(s); fill or drop them first")
        infinite = int(np.isinf(series).sum())
        if infinite:
            raise ValueError(f"linear_regression() {label} has {infinite} infinite value(s)")

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
    # None only failed later, inside np.dot on the first window.
    _require_input(w, "weights", "w")

    def _dot(x: Any) -> Any:
        return np.dot(w, x)

    return _dot


def zero(x: float) -> float:
    """If the value is close to zero, then return zero. Otherwise return itself."""
    _require_input(x, "zero", "x")
    return 0 if abs(x) < sflt.epsilon else x


def df_error_analysis(dfA: DataFrame, dfB: DataFrame, *, corr_method: str = "pearson", plot: bool = False, triangular: bool = False) -> DataFrame:
    """Correlation between two DataFrames, used by the test suite for oracle parity checks."""
    _require_input(dfA, "df_error_analysis", "dfA")
    _require_input(dfB, "df_error_analysis", "dfB")
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


def _is_flat(series: Series) -> bool:
    """True when *series* is constant up to the float residue at its own scale.

    degenerate_zero's absolute 1e-12 is a residue at price scale: two values one
    ULP apart at 1e6 leave a deviation of ~8e-11, which it reads as a real spread.
    """
    return bool(degenerate_zero(series.std(), atol=1e-14 * np.abs(series).max()))


def _linear_regression_np(x: Series, y: Series) -> dict:
    """Simple Linear Regression using Numpy for two 1d arrays."""
    result = {"a": np.nan, "b": np.nan, "r": np.nan, "t": np.nan, "line": np.nan}

    # A constant x (no variance) makes correlation and slope undefined; skip it.
    # The previous guard ``int(x_sum) != 0`` also skipped x whose values summed
    # to less than 1 in absolute value (daily benchmark returns), so the
    # regression never ran for Jensen's alpha. ``x.std() != 0`` then missed the
    # float residue a flat 0.3 leaves (~4e-17), so the slope divided by it and
    # read +-inf, with a correlation made of rounding noise. The tolerance scales
    # with x: at 1e6 the residue is ~8e-11 and the slope read -1.8e8.
    if _is_flat(x):
        return result

    # Centred sums, not m * sum(x * y) - sum(x) * sum(y): that one-pass form
    # cancels catastrophically when x sits far from 0 with a small spread. At
    # x ~ 1e6 with a 1e-3 spread the slope was 99% off, at 1e8 its sign flipped.
    m = x.size
    x_mean, y_mean = x.mean(), y.mean()
    dx = x - x_mean
    b = (dx * (y - y_mean)).sum() / (dx * dx).sum()
    a = y_mean - b * x_mean
    line = a + b * x

    # A constant y still has a slope (0) and an intercept (its level), but no
    # correlation: np.corrcoef divides by y's deviation and warns.
    r = t = np.nan
    if not _is_flat(y):
        # 1st row, 2nd col value corr(x, y)
        r = np.corrcoef(x, y)[0, 1]
        # |r| == 1 or m == 2 divides by zero; that reads inf by design. A local
        # errstate, not np.seterr: the global setting leaked if anything raised.
        with np.errstate(divide="ignore", invalid="ignore"):
            t = r / np.sqrt((1 - r * r) / (m - 2))

    return {"a": a, "b": b, "r": r, "t": t, "line": line}
