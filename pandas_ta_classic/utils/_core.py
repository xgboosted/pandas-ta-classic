# -*- coding: utf-8 -*-
from typing import Any, List, Optional, Tuple, Union

import re as re_
from sys import float_info as sflt

from numpy import argmax, argmin
from pandas import DataFrame, Series
from pandas.api.types import is_datetime64_any_dtype
from pandas_ta_classic import Imports


def _camelCase2Title(x: str) -> str:
    """https://stackoverflow.com/questions/5020906/python-convert-camel-case-to-space-delimited-using-regex-and-taking-acronyms-in"""
    return re_.sub("([a-z])([A-Z])", r"\g<1> \g<2>", x).title()


def get_drift(x: Optional[int]) -> int:
    """Returns an int if not zero, otherwise defaults to one."""
    return int(x) if isinstance(x, int) and x != 0 else 1


def get_offset(x: Optional[int]) -> int:
    """Returns an int, otherwise defaults to zero."""
    return int(x) if isinstance(x, int) else 0


def is_datetime_ordered(df: Union[DataFrame, Series]) -> bool:
    """Returns True if the index is a datetime and ordered."""
    if not is_datetime64_any_dtype(df.index):
        return False
    try:
        return df.index[0] < df.index[-1]
    except Exception:
        return False


def is_percent(x: Optional[Union[int, float]]) -> bool:
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


def tal_ma(name: str) -> Any:
    """Helper Function that returns the Enum value for TA Lib's MA Type"""
    if Imports["talib"] and isinstance(name, str) and len(name) > 1:
        from talib import MA_Type

        _map = {
            "sma": MA_Type.SMA,  # 0
            "ema": MA_Type.EMA,  # 1
            "wma": MA_Type.WMA,  # 2
            "dema": MA_Type.DEMA,  # 3
            "tema": MA_Type.TEMA,  # 4
            "trima": MA_Type.TRIMA,  # 5
            "kama": MA_Type.KAMA,  # 6
            "mama": MA_Type.MAMA,  # 7
            "t3": MA_Type.T3,  # 8
        }
        return _map.get(name.lower(), 0)
    return 0  # Default: SMA -> 0


def unsigned_differences(
    series: Series, amount: Optional[int] = None, **kwargs: Any
) -> Tuple[Series, Series]:
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


def _sma_seed(series: Series, length: int) -> Series:
    """Return a copy of *series* with ``NaN`` before position *length-1* and
    the SMA of the first *length* values at position *length-1*.

    Used by EMA/RMA to initialise exponential smoothing with an SMA seed.
    """
    import numpy as np

    s = series.copy()
    sma_val = s.iloc[:length].mean()
    s.iloc[: length - 1] = np.nan
    s.iloc[length - 1] = sma_val
    return s


def _sliding_weighted_ma(close: Series, length: int, weights: Any) -> Series:
    """Vectorised weighted MA via :func:`sliding_window_view`.

    Args:
        close: The input series.
        length: Window length (must equal ``len(weights)``).
        weights: 1-D weight array whose orientation matches the window layout
            (oldest-first unless caller reverses it).

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


def _get_tal_mode(talib: Any) -> bool:
    """Return True unless *talib* is explicitly ``False``."""
    return bool(talib) if isinstance(talib, bool) else True


def _get_min_periods(kwargs: dict, default: int, key: str = "min_periods") -> int:
    """Extract min_periods from kwargs, defaulting to *default*."""
    v = kwargs.get(key)
    return int(v) if v is not None else default


def _swap_fast_slow(fast: int, slow: int) -> Tuple[int, int]:
    """Ensure ``fast <= slow``; swap if necessary."""
    return (slow, fast) if slow < fast else (fast, slow)


def _build_dataframe(
    series_map: dict,
    name: str,
    category: str,
    offset: int = 0,
    **kwargs: Any,
) -> DataFrame:
    """Build a named/categorized DataFrame from a dict of named Series.

    Applies :func:`apply_offset` to each Series, sets ``name`` and
    ``category`` on both the individual Series and the resulting DataFrame.

    Args:
        series_map: ``{column_name: series}`` mapping.
        name: The DataFrame name (e.g. ``"BBANDS_5_2.0"``).
        category: The indicator category (e.g. ``"volatility"``).
        offset: Number of periods to shift each series.
        **kwargs: Passed through to :func:`apply_offset`.

    Returns:
        A named and categorized DataFrame.

    Example:
        >>> return _build_dataframe(
        ...     {f"DCL_{l}": lower, f"DCM_{l}": mid, f"DCU_{l}": upper},
        ...     f"DC_{l}", "volatility", offset, **kwargs,
        ... )
    """
    data = {}
    for col_name, s in series_map.items():
        s = apply_offset(s, offset, **kwargs)
        s.name = col_name
        s.category = category
        data[col_name] = s
    df = DataFrame(data)
    df.name = name
    df.category = category
    return df


def _finalize(
    result: Union[Series, DataFrame],
    offset: int,
    name: str,
    category: str,
    **kwargs: Any,
) -> Union[Series, DataFrame]:
    """Apply offset, fill, and set name/category on a result Series or DataFrame.

    Args:
        result: The computed indicator result.
        offset: Number of periods to shift.
        name: The indicator name (e.g. ``"SMA_10"``).
        category: The indicator category (e.g. ``"overlap"``).
        **kwargs: Passed through to :func:`apply_offset` (``fillna``, ``fill_method``).

    Returns:
        The finalized Series or DataFrame.

    Example:
        >>> return _finalize(sma, offset, f"SMA_{length}", "overlap", **kwargs)
    """
    result = apply_offset(result, offset, **kwargs)
    result.name = name
    result.category = category
    return result


def apply_offset(
    result: Union[Series, DataFrame],
    offset: int,
    **kwargs: Any,
) -> Union[Series, DataFrame]:
    """Apply offset shift and optional fill operations to a Series or DataFrame.

    Args:
        result: A pandas Series or DataFrame to modify.
        offset: Number of periods to shift. 0 means no shift.
        **kwargs: Supports ``fillna`` (scalar) and ``fill_method``
            (``"ffill"`` or ``"bfill"``).

    Returns:
        The modified Series or DataFrame (also returned for chaining).

    Example:
        >>> rsi = apply_offset(rsi, offset, **kwargs)
    """
    if offset != 0:
        result = result.shift(offset)
    if "fillna" in kwargs:
        result.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        fm = kwargs["fill_method"]
        if fm == "ffill":
            result.ffill(inplace=True)
        elif fm == "bfill":
            result.bfill(inplace=True)
    return result


def verify_series(
    series: Series, min_length: Optional[Union[int, float]] = None
) -> Optional[Series]:
    """If a Pandas Series and it meets the min_length of the indicator return it."""
    has_length = min_length is not None and isinstance(min_length, int)
    if series is not None and isinstance(series, Series):
        return None if has_length and series.size < min_length else series
    return None
