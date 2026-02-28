# -*- coding: utf-8 -*-
from typing import Any, List, Optional, Tuple, Union

import re as re_
from pathlib import Path
from sys import float_info as sflt

from numpy import argmax, argmin
from pandas import DataFrame, Series
from pandas.api.types import is_datetime64_any_dtype
from pandas_ta_classic import Imports


def _camelCase2Title(x: str) -> str:
    """https://stackoverflow.com/questions/5020906/python-convert-camel-case-to-space-delimited-using-regex-and-taking-acronyms-in"""
    return re_.sub("([a-z])([A-Z])", r"\g<1> \g<2>", x).title()


def category_files(category: str) -> List[str]:
    """Helper function to return all filenames in the category directory."""
    files = [
        x.stem
        for x in list(Path(f"pandas_ta_classic/{category}/").glob("*.py"))
        if x.stem != "__init__"
    ]
    return files


def get_drift(x: Optional[int]) -> int:
    """Returns an int if not zero, otherwise defaults to one."""
    return int(x) if isinstance(x, int) and x != 0 else 1


def get_offset(x: Optional[int]) -> int:
    """Returns an int, otherwise defaults to zero."""
    return int(x) if isinstance(x, int) else 0


def is_datetime_ordered(df: Union[DataFrame, Series]) -> bool:
    """Returns True if the index is a datetime and ordered."""
    index_is_datetime = is_datetime64_any_dtype(df.index)
    try:
        ordered = df.index[0] < df.index[-1]
    except RuntimeWarning:
        pass
    finally:
        return True if index_is_datetime and ordered else False


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

        name = name.lower()
        if name == "sma":
            return MA_Type.SMA  # 0
        elif name == "ema":
            return MA_Type.EMA  # 1
        elif name == "wma":
            return MA_Type.WMA  # 2
        elif name == "dema":
            return MA_Type.DEMA  # 3
        elif name == "tema":
            return MA_Type.TEMA  # 4
        elif name == "trima":
            return MA_Type.TRIMA  # 5
        elif name == "kama":
            return MA_Type.KAMA  # 6
        elif name == "mama":
            return MA_Type.MAMA  # 7
        elif name == "t3":
            return MA_Type.T3  # 8
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


def apply_offset(
    series: "Series",
    offset: int,
    **kwargs: Any,
) -> "Series":
    """Apply offset shift and optional fill operations to a Series or DataFrame.

    Args:
        series: A pandas Series or DataFrame to modify.
        offset: Number of periods to shift. 0 means no shift.
        **kwargs: Supports ``fillna`` (scalar) and ``fill_method``
            (``"ffill"`` or ``"bfill"``).

    Returns:
        The modified series (also returned for chaining).

    Example:
        >>> rsi = apply_offset(rsi, offset, **kwargs)
    """
    if offset != 0:
        series = series.shift(offset)
    if "fillna" in kwargs:
        series.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        fm = kwargs["fill_method"]
        if fm == "ffill":
            series.ffill(inplace=True)
        elif fm == "bfill":
            series.bfill(inplace=True)
    return series


def verify_series(
    series: Series, min_length: Optional[Union[int, float]] = None
) -> Optional[Series]:
    """If a Pandas Series and it meets the min_length of the indicator return it."""
    has_length = min_length is not None and isinstance(min_length, int)
    if series is not None and isinstance(series, Series):
        return None if has_length and series.size < min_length else series
    return None
