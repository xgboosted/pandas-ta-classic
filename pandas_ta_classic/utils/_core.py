import logging
from typing import Any, Optional, TypeGuard, Union

from sys import float_info as sflt

from numpy import argmax, argmin
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
    series: Union[Series, DataFrame, list[Union[Series, DataFrame]]],
    offset: int = 0,
) -> Union[Series, DataFrame, list[Union[Series, DataFrame]]]:
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
    series: Union[Series, DataFrame, list[Union[Series, DataFrame]]],
    **kwargs: Any,
) -> Union[Series, DataFrame, list[Union[Series, DataFrame]]]:
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


def get_drift(x: Optional[int]) -> int:
    """Returns an int if not zero, otherwise defaults to one."""
    return int(x) if isinstance(x, int) and x != 0 else 1


def get_offset(x: Optional[int]) -> int:
    """Returns an int, otherwise defaults to zero."""
    return int(x) if isinstance(x, int) else 0


def is_datetime_ordered(df: Union[DataFrame, Series]) -> bool:
    """Returns True if the index is a datetime and ordered."""
    index_is_datetime = is_datetime64_any_dtype(df.index)
    if not index_is_datetime or len(df.index) < 2:
        return False
    try:
        return bool(df.index[0] < df.index[-1])
    except (IndexError, TypeError):
        return False


def is_percent(x: Optional[Union[int, float]]) -> TypeGuard[Union[int, float]]:
    if isinstance(x, (int, float)):
        return x is not None and x >= 0 and x <= 100
    return False


def non_zero_range(high: Series, low: Series) -> Series:
    """Returns the difference of two series and adds epsilon to any zero values.  This occurs commonly in crypto data when 'high' = 'low'."""
    diff = high - low
    if diff.eq(0).any().any():
        diff += sflt.epsilon
    return diff


def recent_maximum_index(x: Series) -> int:
    return int(argmax(x[::-1]))


def recent_minimum_index(x: Series) -> int:
    return int(argmin(x[::-1]))


def signed_series(series: Series, initial: Optional[int] = None) -> Series:
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


def unsigned_differences(series: Series, amount: Optional[int] = None, **kwargs: Any) -> tuple[Series, Series]:
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


def verify_series(series: Series, min_length: Optional[Union[int, float]] = None) -> Optional[Series]:
    """If a Pandas Series and it meets the min_length of the indicator return it."""
    has_length = min_length is not None and isinstance(min_length, int)
    if series is not None and isinstance(series, Series):
        if has_length and series.size < min_length:
            logger.warning(f"[X] Series has {series.size} rows but indicator requires" f" at least {min_length}. Returning None.")
            return None
        return series
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
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view

    arr = close.to_numpy(dtype=float)
    windows = sliding_window_view(arr, length)
    result = np.full(len(arr), np.nan)
    result[length - 1 :] = windows @ weights
    return Series(result, index=close.index)
