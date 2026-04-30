# -*- coding: utf-8 -*-
"""Math Operators and Transforms for pandas-ta-classic.

Covers TA-Lib's Math Operator (ADD, SUB, DIV, MULT, MAX, MIN, SUM,
MAXINDEX, MININDEX, MINMAX, MINMAXINDEX) and Math Transform (ACOS, ASIN,
ATAN, CEIL, COS, COSH, EXP, FLOOR, LN, LOG10, SIN, SINH, SQRT, TAN, TANH)
groups, plus tulipy extras (ABS, ROUND, TRUNC, TODEG, TORAD).
"""

from typing import Any, Optional

import numpy as np
from pandas import DataFrame, Series

from pandas_ta_classic.utils import get_offset, verify_series

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_two(a: Series, b: Series):
    a = verify_series(a)
    b = verify_series(b)
    return a, b


# ---------------------------------------------------------------------------
# Math Operators – element-wise two-series
# ---------------------------------------------------------------------------


def add(
    series_a: Series,
    series_b: Series,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Element-wise addition of two Series (TA-Lib: ADD)."""
    series_a, series_b = _validate_two(series_a, series_b)
    if series_a is None or series_b is None:
        return None
    offset = get_offset(offset)
    result = series_a + series_b
    if offset != 0:
        result = result.shift(offset)
    result.name = "ADD"
    result.category = "math"
    return result


def sub(
    series_a: Series,
    series_b: Series,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Element-wise subtraction of two Series (TA-Lib: SUB)."""
    series_a, series_b = _validate_two(series_a, series_b)
    if series_a is None or series_b is None:
        return None
    offset = get_offset(offset)
    result = series_a - series_b
    if offset != 0:
        result = result.shift(offset)
    result.name = "SUB"
    result.category = "math"
    return result


def div(
    series_a: Series,
    series_b: Series,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Element-wise division of two Series (TA-Lib: DIV)."""
    series_a, series_b = _validate_two(series_a, series_b)
    if series_a is None or series_b is None:
        return None
    offset = get_offset(offset)
    result = series_a / series_b
    if offset != 0:
        result = result.shift(offset)
    result.name = "DIV"
    result.category = "math"
    return result


def mult(
    series_a: Series,
    series_b: Series,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Element-wise multiplication of two Series (TA-Lib: MULT)."""
    series_a, series_b = _validate_two(series_a, series_b)
    if series_a is None or series_b is None:
        return None
    offset = get_offset(offset)
    result = series_a * series_b
    if offset != 0:
        result = result.shift(offset)
    result.name = "MULT"
    result.category = "math"
    return result


# ---------------------------------------------------------------------------
# Math Operators – rolling window
# ---------------------------------------------------------------------------


def rolling_max(
    close: Series,
    length: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Rolling Maximum over *length* periods (TA-Lib: MAX)."""
    length = int(length) if length and length > 0 else 30
    close = verify_series(close, length)
    offset = get_offset(offset)
    if close is None:
        return None
    result = close.rolling(length).max()
    if offset != 0:
        result = result.shift(offset)
    result.name = f"MAX_{length}"
    result.category = "math"
    return result


# Public alias matching TA-Lib name
max = rolling_max  # noqa: A001


def rolling_min(
    close: Series,
    length: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Rolling Minimum over *length* periods (TA-Lib: MIN)."""
    length = int(length) if length and length > 0 else 30
    close = verify_series(close, length)
    offset = get_offset(offset)
    if close is None:
        return None
    result = close.rolling(length).min()
    if offset != 0:
        result = result.shift(offset)
    result.name = f"MIN_{length}"
    result.category = "math"
    return result


min = rolling_min  # noqa: A001


def rolling_sum(
    close: Series,
    length: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Rolling Summation over *length* periods (TA-Lib: SUM)."""
    length = int(length) if length and length > 0 else 30
    close = verify_series(close, length)
    offset = get_offset(offset)
    if close is None:
        return None
    result = close.rolling(length).sum()
    if offset != 0:
        result = result.shift(offset)
    result.name = f"SUM_{length}"
    result.category = "math"
    return result


sum = rolling_sum  # noqa: A001


def maxindex(
    close: Series,
    length: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Index of the Maximum value over *length* periods (TA-Lib: MAXINDEX).

    Returns 0-based index within the rolling window.
    """
    length = int(length) if length and length > 0 else 30
    close = verify_series(close, length)
    offset = get_offset(offset)
    if close is None:
        return None
    result = close.rolling(length).apply(np.argmax, raw=True)
    if offset != 0:
        result = result.shift(offset)
    result.name = f"MAXINDEX_{length}"
    result.category = "math"
    return result


def minindex(
    close: Series,
    length: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Index of the Minimum value over *length* periods (TA-Lib: MININDEX).

    Returns 0-based index within the rolling window.
    """
    length = int(length) if length and length > 0 else 30
    close = verify_series(close, length)
    offset = get_offset(offset)
    if close is None:
        return None
    result = close.rolling(length).apply(np.argmin, raw=True)
    if offset != 0:
        result = result.shift(offset)
    result.name = f"MININDEX_{length}"
    result.category = "math"
    return result


def minmax(
    close: Series,
    length: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Rolling Min and Max over *length* periods (TA-Lib: MINMAX).

    Returns a DataFrame with columns ``MIN_<n>`` and ``MAX_<n>``.
    """
    length = int(length) if length and length > 0 else 30
    close = verify_series(close, length)
    offset = get_offset(offset)
    if close is None:
        return None
    mn = close.rolling(length).min()
    mx = close.rolling(length).max()
    if offset != 0:
        mn = mn.shift(offset)
        mx = mx.shift(offset)
    mn.name = f"MIN_{length}"
    mx.name = f"MAX_{length}"
    df = DataFrame({mn.name: mn, mx.name: mx})
    df.name = f"MINMAX_{length}"
    df.category = "math"
    return df


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
    if offset != 0:
        mn_idx = mn_idx.shift(offset)
        mx_idx = mx_idx.shift(offset)
    mn_idx.name = f"MINIDX_{length}"
    mx_idx.name = f"MAXIDX_{length}"
    df = DataFrame({mn_idx.name: mn_idx, mx_idx.name: mx_idx})
    df.name = f"MINMAXINDEX_{length}"
    df.category = "math"
    return df


# ---------------------------------------------------------------------------
# Math Transforms – element-wise numpy wrappers
# ---------------------------------------------------------------------------


def _transform(close, fn, name, offset):
    close = verify_series(close)
    if close is None:
        return None
    offset = get_offset(offset)
    result = close.apply(fn)
    if offset != 0:
        result = result.shift(offset)
    result.name = name
    result.category = "math"
    return result


def acos(close: Series, offset: Optional[int] = None, **kwargs) -> Optional[Series]:
    """Vector Trigonometric ACos (TA-Lib: ACOS)."""
    return _transform(close, np.arccos, "ACOS", offset)


def asin(close: Series, offset: Optional[int] = None, **kwargs) -> Optional[Series]:
    """Vector Trigonometric ASin (TA-Lib: ASIN)."""
    return _transform(close, np.arcsin, "ASIN", offset)


def atan(close: Series, offset: Optional[int] = None, **kwargs) -> Optional[Series]:
    """Vector Trigonometric ATan (TA-Lib: ATAN)."""
    return _transform(close, np.arctan, "ATAN", offset)


def ceil(close: Series, offset: Optional[int] = None, **kwargs) -> Optional[Series]:
    """Vector Ceil (TA-Lib: CEIL)."""
    return _transform(close, np.ceil, "CEIL", offset)


def cos(close: Series, offset: Optional[int] = None, **kwargs) -> Optional[Series]:
    """Vector Trigonometric Cos (TA-Lib: COS)."""
    return _transform(close, np.cos, "COS", offset)


def cosh(close: Series, offset: Optional[int] = None, **kwargs) -> Optional[Series]:
    """Vector Trigonometric Cosh (TA-Lib: COSH)."""
    return _transform(close, np.cosh, "COSH", offset)


def exp(close: Series, offset: Optional[int] = None, **kwargs) -> Optional[Series]:
    """Vector Arithmetic Exp (TA-Lib: EXP)."""
    return _transform(close, np.exp, "EXP", offset)


def floor(close: Series, offset: Optional[int] = None, **kwargs) -> Optional[Series]:
    """Vector Floor (TA-Lib: FLOOR)."""
    return _transform(close, np.floor, "FLOOR", offset)


def ln(close: Series, offset: Optional[int] = None, **kwargs) -> Optional[Series]:
    """Vector Log Natural (TA-Lib: LN)."""
    return _transform(close, np.log, "LN", offset)


def log10(close: Series, offset: Optional[int] = None, **kwargs) -> Optional[Series]:
    """Vector Log10 (TA-Lib: LOG10)."""
    return _transform(close, np.log10, "LOG10", offset)


def sin(close: Series, offset: Optional[int] = None, **kwargs) -> Optional[Series]:
    """Vector Trigonometric Sin (TA-Lib: SIN)."""
    return _transform(close, np.sin, "SIN", offset)


def sinh(close: Series, offset: Optional[int] = None, **kwargs) -> Optional[Series]:
    """Vector Trigonometric Sinh (TA-Lib: SINH)."""
    return _transform(close, np.sinh, "SINH", offset)


def sqrt(close: Series, offset: Optional[int] = None, **kwargs) -> Optional[Series]:
    """Vector Square Root (TA-Lib: SQRT)."""
    return _transform(close, np.sqrt, "SQRT", offset)


def tan(close: Series, offset: Optional[int] = None, **kwargs) -> Optional[Series]:
    """Vector Trigonometric Tan (TA-Lib: TAN)."""
    return _transform(close, np.tan, "TAN", offset)


def tanh(close: Series, offset: Optional[int] = None, **kwargs) -> Optional[Series]:
    """Vector Trigonometric Tanh (TA-Lib: TANH)."""
    return _transform(close, np.tanh, "TANH", offset)


# ---------------------------------------------------------------------------
# tulipy extras – element-wise
# ---------------------------------------------------------------------------


def npabs(close: Series, offset: Optional[int] = None, **kwargs) -> Optional[Series]:
    """Vector Absolute Value (tulipy: ABS)."""
    return _transform(close, np.abs, "ABS", offset)


def npround(close: Series, offset: Optional[int] = None, **kwargs) -> Optional[Series]:
    """Vector Round (tulipy: ROUND)."""
    return _transform(close, np.round, "ROUND", offset)


def trunc(close: Series, offset: Optional[int] = None, **kwargs) -> Optional[Series]:
    """Vector Truncate (tulipy: TRUNC)."""
    return _transform(close, np.trunc, "TRUNC", offset)


def todeg(close: Series, offset: Optional[int] = None, **kwargs) -> Optional[Series]:
    """Vector Degrees conversion (tulipy: TODEG). Converts radians to degrees."""
    return _transform(close, np.degrees, "TODEG", offset)


def torad(close: Series, offset: Optional[int] = None, **kwargs) -> Optional[Series]:
    """Vector Radians conversion (tulipy: TORAD). Converts degrees to radians."""
    return _transform(close, np.radians, "TORAD", offset)
