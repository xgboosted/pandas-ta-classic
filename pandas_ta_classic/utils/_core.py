import functools
import inspect
import logging
import sys
from collections.abc import Callable
from sys import float_info as sflt
from typing import Any, TypeGuard

import numpy as np
from pandas import DataFrame, Series
from pandas.api.types import is_datetime64_any_dtype

logger = logging.getLogger(__name__)


def _pos_int(val, default):
    """Return ``int(val)`` when *val* is a positive integer, else *default*."""
    return int(val) if val and val > 0 else default


def _pos_float(val, default):
    """Return ``float(val)`` when *val* is a positive float, else *default*."""
    return float(val) if val and val > 0 else default


def apply_offset(
    series: Series | DataFrame | list[Series | DataFrame],
    offset: int = 0,
) -> Series | DataFrame | list[Series | DataFrame]:
    """Shift one or more Series/DataFrames by *offset* periods.

    Args:
        series: A single Series/DataFrame, or a list of them.
        offset: Number of periods to shift. ``0`` means no shift.

    Returns:
        The shifted object(s), same type structure as *series*.
    """
    if isinstance(series, (list, tuple)):
        return [apply_offset(s, offset) for s in series]
    return series.shift(offset) if offset != 0 else series


def apply_fill(
    series: Series | DataFrame | list[Series | DataFrame],
    **kwargs: Any,
) -> Series | DataFrame | list[Series | DataFrame]:
    """Apply fillna and fill_method from kwargs to one or more Series/DataFrames.

    Args:
        series: A single Series/DataFrame, or a list of them.
        **kwargs: Recognised keys:
            ``fillna`` -- value passed to ``Series.fillna()``.
            ``fill_method`` -- ``"ffill"`` or ``"bfill"``.

    Returns:
        The processed object(s), same type structure as *series*.
    """
    if isinstance(series, (list, tuple)):
        return [apply_fill(s, **kwargs) for s in series]
    if "fillna" in kwargs:
        series.fillna(kwargs["fillna"], inplace=True)
    fill_method = kwargs.get("fill_method")
    if fill_method == "ffill":
        series.ffill(inplace=True)
    elif fill_method == "bfill":
        series.bfill(inplace=True)
    return series


def get_drift(x: int | None) -> int:
    """Returns an int if not zero, otherwise defaults to one."""
    return int(x) if isinstance(x, int) and x != 0 else 1


def get_offset(x: int | None) -> int:
    """Returns an int, otherwise defaults to zero."""
    return int(x) if isinstance(x, int) else 0


def is_datetime_ordered(df: DataFrame | Series) -> bool:
    """Returns True if the index is a datetime and ordered."""
    index_is_datetime = is_datetime64_any_dtype(df.index)
    if not index_is_datetime or len(df.index) < 2:
        return False
    try:
        return bool(df.index[0] < df.index[-1])
    except (IndexError, TypeError):
        return False


def is_percent(x: float | None) -> TypeGuard[float]:
    if isinstance(x, (int, float)):
        return x is not None and x >= 0 and x <= 100
    return False


def leading_nan_rows(*series: Series) -> int:
    """Number of leading rows before every series has a finite value."""
    finite = np.isfinite(np.vstack([s.to_numpy(dtype=float) for s in series])).all(axis=0)
    hits = np.flatnonzero(finite)
    return int(hits[0]) if hits.size else finite.size


def skip_leading_nan(*names: str) -> Callable:
    """Compute an indicator on the rows after a leading NaN run, then pad back.

    Chained input (another indicator's output) always starts with NaN. A
    recursion seeded from the first bar carries that NaN forward forever, so
    the whole result came back NaN. The decorated function instead receives
    every Series argument trimmed to start at the first row where all of
    *names* are finite; its result is reindexed to the original index, with
    ``name`` and ``category`` preserved. Input without a leading NaN run, or
    that is entirely NaN, is passed through unchanged.
    """

    def decorator(fn: Callable) -> Callable:
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            bound = sig.bind_partial(*args, **kwargs)
            primary = [bound.arguments.get(n) for n in names]
            if not all(isinstance(s, Series) for s in primary) or len({s.size for s in primary}) != 1:
                return fn(*args, **kwargs)
            size = primary[0].size
            start = leading_nan_rows(*primary)
            if not 0 < start < size:
                return fn(*args, **kwargs)
            for key, value in bound.arguments.items():
                if isinstance(value, Series) and value.size == size:
                    bound.arguments[key] = value.iloc[start:]
            result = fn(*bound.args, **bound.kwargs)
            if result is None:
                return None
            padded = result.reindex(primary[0].index)
            for attr in ("name", "category"):
                if hasattr(result, attr):
                    setattr(padded, attr, getattr(result, attr))
            return padded

        return wrapper

    return decorator


def non_zero_range(high: Series, low: Series) -> Series:
    """Returns the difference of two series, replacing exact zeros with epsilon.  This occurs commonly in crypto data when 'high' = 'low'.

    The replacement is pointwise: a flat bar at row ``t`` only affects row ``t``.
    Rows with a non-zero range keep their exact difference, so the value at row
    ``t`` never depends on bars at ``t + 1`` or later.
    """
    diff = high - low
    return diff.where(diff != 0, sflt.epsilon)


def recent_maximum_index(x: Series) -> int:
    return int(np.argmax(x[::-1]))


def recent_minimum_index(x: Series) -> int:
    return int(np.argmin(x[::-1]))


def signed_series(series: Series, initial: int | None = None) -> Series:
    """Returns a Signed Series with or without an initial value

    Default Example:
    series = Series([3, 2, 2, 1, 1, 5, 6, 6, 7, 5])
    and returns:
    sign = Series([NaN, -1.0, 0.0, -1.0, 0.0, 1.0, 1.0, 0.0, 1.0, -1.0])
    """
    series = verify_series(series)
    sign = series.diff(1)
    sign[sign > 0] = 1
    sign[sign < 0] = -1
    sign.iloc[0] = initial
    return sign


# TA-Lib MA_Type enum values are frozen ABI constants, so they are mapped
# directly here rather than importing talib just to read them. This keeps
# tal_ma usable (and testable) without the optional talib dependency.
_TAL_MA_TYPES = {
    "sma": 0,
    "ema": 1,
    "wma": 2,
    "dema": 3,
    "tema": 4,
    "trima": 5,
    "kama": 6,
    "mama": 7,
    "t3": 8,
}


def tal_ma(name: str) -> int:
    """Return the TA-Lib MA_Type enum value for an MA name (``sma``..``t3``).

    Raises:
        TypeError: if *name* is not a string.
        ValueError: if *name* is not a recognised TA-Lib MA type.
    """
    if not isinstance(name, str):
        raise TypeError(f"tal_ma expects a str MA name, got {type(name).__name__}")
    key = name.lower()
    if key not in _TAL_MA_TYPES:
        raise ValueError(f"Unknown TA-Lib MA type {name!r}; valid: {sorted(_TAL_MA_TYPES)}")
    return _TAL_MA_TYPES[key]


def unsigned_differences(series: Series, amount: int | None = None, **kwargs: Any) -> tuple[Series, Series]:
    """Unsigned Differences
    Returns two Series, an unsigned positive and unsigned negative series based
    on the differences of the original series. The positive series are only the
    increases and the negative series are only the decreases.

    Default Example:
    series   = Series([3, 2, 2, 1, 1, 5, 6, 6, 7, 5, 3]) and returns
    postive  = Series([0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0])
    negative = Series([0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1])
    """
    amount = int(amount) if amount is not None else 1
    negative = series.diff(amount)
    negative.fillna(0, inplace=True)
    positive = negative.copy()

    positive[positive <= 0] = 0
    positive[positive > 0] = 1

    negative[negative >= 0] = 0
    negative[negative < 0] = 1

    if kwargs.pop("asint", False):
        positive = positive.astype(int)
        negative = negative.astype(int)

    return positive, negative


def verify_series(series: Series, min_length: float | None = None) -> Series | None:
    """If a Pandas Series and it meets the min_length of the indicator return it.

    Returns None for a Series shorter than *min_length* (an ordinary data
    condition) and for ``None`` (an optional argument that was not given).

    Anything else -- a list, a numpy array, a DataFrame -- is a caller error
    and raises TypeError, so the mistake surfaces where it was made rather than
    as a missing column or an unrelated error several frames later.
    """
    has_length = min_length is not None and isinstance(min_length, int)
    if series is not None and isinstance(series, Series):
        if has_length and series.size < min_length:
            logger.warning(f"[X] Series has {series.size} rows but indicator requires" f" at least {min_length}. Returning None.")
            return None
        return series
    if series is not None:
        indicator = sys._getframe(1).f_code.co_name  # the indicator that called verify_series
        raise TypeError(
            f"{indicator}() expected a pandas Series but got {type(series).__name__}. "
            "Pass a Series, e.g. df['close'] rather than df['close'].values."
        )
    return None


def _sliding_weighted_ma(close: Series, length: int, weights: Any) -> Series:
    """Vectorised weighted MA via sliding_window_view.

    Args:
        close: The input series.
        length: Window length (must equal ``len(weights)``).
        weights: 1-D weight array whose orientation matches the window layout.

    Returns:
        A Series aligned with *close*, with ``NaN`` for the first
        ``length - 1`` positions.
    """
    arr = close.to_numpy(dtype=float)
    result = np.full(len(arr), np.nan)
    if length <= arr.shape[0]:
        windows = np.lib.stride_tricks.sliding_window_view(arr, length)
        result[length - 1 :] = windows @ weights
    return Series(result, index=close.index)


def _sliding_argextreme(series: Series, length: int, argfunc: Any, reverse: bool = False) -> Series:
    """Vectorised rolling ``argfunc`` (arg-position of a window extreme).

    Bit-identical to ``series.rolling(length).apply(argfunc, raw=True)`` — an
    integer arg-position over each window, with ``NaN`` for the first
    ``length - 1`` warm-up bars. With ``reverse=True`` the window is flipped
    first, matching ``argfunc(x[::-1])`` (periods since the most-recent extreme).

    Args:
        series: The input series.
        length: Rolling window length.
        argfunc: ``np.argmax`` or ``np.argmin``.
        reverse: Flip each window before applying *argfunc*.
    """
    arr = series.to_numpy(dtype=float)
    m = arr.shape[0]
    result = np.full(m, np.nan)
    if length <= m:
        windows = np.lib.stride_tricks.sliding_window_view(arr, length)
        if reverse:
            windows = windows[:, ::-1]
        result[length - 1 :] = argfunc(windows, axis=1)
    return Series(result, index=series.index)
