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

from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series

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
    result = apply_offset(result, offset)
    result = apply_fill(result, **kwargs)
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
    result = apply_offset(result, offset)
    result = apply_fill(result, **kwargs)
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
    result = apply_offset(result, offset)
    result = apply_fill(result, **kwargs)
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
    result = apply_offset(result, offset)
    result = apply_fill(result, **kwargs)
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
    result = apply_offset(result, offset)
    result = apply_fill(result, **kwargs)
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
    result = apply_offset(result, offset)
    result = apply_fill(result, **kwargs)
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
    result = apply_offset(result, offset)
    result = apply_fill(result, **kwargs)
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
    result = apply_offset(result, offset)
    result = apply_fill(result, **kwargs)
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
    result = apply_offset(result, offset)
    result = apply_fill(result, **kwargs)
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
    mn, mx = apply_offset([mn, mx], offset)
    mn, mx = apply_fill([mn, mx], **kwargs)
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
    mn_idx, mx_idx = apply_offset([mn_idx, mx_idx], offset)
    mn_idx, mx_idx = apply_fill([mn_idx, mx_idx], **kwargs)
    mn_idx.name = f"MINIDX_{length}"
    mx_idx.name = f"MAXIDX_{length}"
    df = DataFrame({mn_idx.name: mn_idx, mx_idx.name: mx_idx})
    df.name = f"MINMAXINDEX_{length}"
    df.category = "math"
    return df


# ---------------------------------------------------------------------------
# Math Transforms – element-wise numpy wrappers
# ---------------------------------------------------------------------------


def _transform(close, fn, name, offset, **kwargs):
    close = verify_series(close)
    if close is None:
        return None
    offset = get_offset(offset)
    result = close.apply(fn)
    result = apply_offset(result, offset)
    result = apply_fill(result, **kwargs)
    result.name = name
    result.category = "math"
    return result


def acos(close: Series, offset: Optional[int] = None, **kwargs) -> Optional[Series]:
    """Vector Trigonometric ACos (TA-Lib: ACOS)."""
    return _transform(close, np.arccos, "ACOS", offset, **kwargs)


def asin(close: Series, offset: Optional[int] = None, **kwargs) -> Optional[Series]:
    """Vector Trigonometric ASin (TA-Lib: ASIN)."""
    return _transform(close, np.arcsin, "ASIN", offset, **kwargs)


def atan(close: Series, offset: Optional[int] = None, **kwargs) -> Optional[Series]:
    """Vector Trigonometric ATan (TA-Lib: ATAN)."""
    return _transform(close, np.arctan, "ATAN", offset, **kwargs)


def ceil(close: Series, offset: Optional[int] = None, **kwargs) -> Optional[Series]:
    """Vector Ceil (TA-Lib: CEIL)."""
    return _transform(close, np.ceil, "CEIL", offset, **kwargs)


def cos(close: Series, offset: Optional[int] = None, **kwargs) -> Optional[Series]:
    """Vector Trigonometric Cos (TA-Lib: COS)."""
    return _transform(close, np.cos, "COS", offset, **kwargs)


def cosh(close: Series, offset: Optional[int] = None, **kwargs) -> Optional[Series]:
    """Vector Trigonometric Cosh (TA-Lib: COSH)."""
    return _transform(close, np.cosh, "COSH", offset, **kwargs)


def exp(close: Series, offset: Optional[int] = None, **kwargs) -> Optional[Series]:
    """Vector Arithmetic Exp (TA-Lib: EXP)."""
    return _transform(close, np.exp, "EXP", offset, **kwargs)


def floor(close: Series, offset: Optional[int] = None, **kwargs) -> Optional[Series]:
    """Vector Floor (TA-Lib: FLOOR)."""
    return _transform(close, np.floor, "FLOOR", offset, **kwargs)


def ln(close: Series, offset: Optional[int] = None, **kwargs) -> Optional[Series]:
    """Vector Log Natural (TA-Lib: LN)."""
    return _transform(close, np.log, "LN", offset, **kwargs)


def log10(close: Series, offset: Optional[int] = None, **kwargs) -> Optional[Series]:
    """Vector Log10 (TA-Lib: LOG10)."""
    return _transform(close, np.log10, "LOG10", offset, **kwargs)


def sin(close: Series, offset: Optional[int] = None, **kwargs) -> Optional[Series]:
    """Vector Trigonometric Sin (TA-Lib: SIN)."""
    return _transform(close, np.sin, "SIN", offset, **kwargs)


def sinh(close: Series, offset: Optional[int] = None, **kwargs) -> Optional[Series]:
    """Vector Trigonometric Sinh (TA-Lib: SINH)."""
    return _transform(close, np.sinh, "SINH", offset, **kwargs)


def sqrt(close: Series, offset: Optional[int] = None, **kwargs) -> Optional[Series]:
    """Vector Square Root (TA-Lib: SQRT)."""
    return _transform(close, np.sqrt, "SQRT", offset, **kwargs)


def tan(close: Series, offset: Optional[int] = None, **kwargs) -> Optional[Series]:
    """Vector Trigonometric Tan (TA-Lib: TAN)."""
    return _transform(close, np.tan, "TAN", offset, **kwargs)


def tanh(close: Series, offset: Optional[int] = None, **kwargs) -> Optional[Series]:
    """Vector Trigonometric Tanh (TA-Lib: TANH)."""
    return _transform(close, np.tanh, "TANH", offset, **kwargs)


# ---------------------------------------------------------------------------
# tulipy extras – element-wise
# ---------------------------------------------------------------------------


def npabs(close: Series, offset: Optional[int] = None, **kwargs) -> Optional[Series]:
    """Vector Absolute Value (tulipy: ABS)."""
    return _transform(close, np.abs, "ABS", offset, **kwargs)


def npround(close: Series, offset: Optional[int] = None, **kwargs) -> Optional[Series]:
    """Vector Round (tulipy: ROUND)."""
    return _transform(close, np.round, "ROUND", offset, **kwargs)


def trunc(close: Series, offset: Optional[int] = None, **kwargs) -> Optional[Series]:
    """Vector Truncate (tulipy: TRUNC)."""
    return _transform(close, np.trunc, "TRUNC", offset, **kwargs)


def todeg(close: Series, offset: Optional[int] = None, **kwargs) -> Optional[Series]:
    """Vector Degrees conversion (tulipy: TODEG). Converts radians to degrees."""
    return _transform(close, np.degrees, "TODEG", offset, **kwargs)


def torad(close: Series, offset: Optional[int] = None, **kwargs) -> Optional[Series]:
    """Vector Radians conversion (tulipy: TORAD). Converts degrees to radians."""
    return _transform(close, np.radians, "TORAD", offset, **kwargs)
