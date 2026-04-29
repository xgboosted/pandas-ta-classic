# -*- coding: utf-8 -*-
# Moving Average with Variable Period (MAVP)
import warnings
from typing import Any, Optional

import numpy as np
from pandas import Series

from pandas_ta_classic import Imports
from pandas_ta_classic.utils import get_offset, verify_series


def mavp(
    close: Series,
    periods: Optional[Series] = None,
    minperiod: Optional[int] = None,
    maxperiod: Optional[int] = None,
    mamode: Optional[int] = None,
    talib: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Moving Average with Variable Period (MAVP)"""
    # Validate Arguments
    minperiod = int(minperiod) if minperiod and minperiod >= 2 else 2
    maxperiod = int(maxperiod) if maxperiod and maxperiod > minperiod else 30
    # mamode: 0=SMA, 1=EMA, 2=WMA, 3=DEMA, 4=TEMA, 5=TRIMA, 6=KAMA, 7=MAMA, 8=T3
    # For native fallback we only support SMA (0)
    mamode = int(mamode) if mamode is not None else 0
    close = verify_series(close, maxperiod)
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    if close is None:
        return None

    if periods is None:
        # Default: linearly vary from minperiod to maxperiod
        n = len(close)
        periods = Series(
            np.linspace(minperiod, maxperiod, n), index=close.index, dtype=float
        )
    else:
        periods = verify_series(periods)
        if periods is None:
            return None

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from talib import MAVP as TAMAVP

        mavp_ = TAMAVP(
            close,
            periods.astype(float),
            minperiod=minperiod,
            maxperiod=maxperiod,
            matype=mamode,
        )
    else:
        # Native: simple moving average with per-bar variable window
        # Only SMA (mamode=0) is supported natively; other MA types require TA-Lib
        if mamode != 0:
            warnings.warn(
                f"MAVP native fallback only supports SMA (mamode=0); "
                f"mamode={mamode} requires TA-Lib. Results will use SMA.",
                UserWarning,
                stacklevel=2,
            )
        close_arr = close.to_numpy(dtype=float)
        per_arr = np.clip(
            periods.to_numpy(dtype=float).round().astype(int), minperiod, maxperiod
        )
        n = len(close_arr)
        result = np.full(n, np.nan)
        for i in range(n):
            p = per_arr[i]
            if i + 1 >= p:
                result[i] = close_arr[i - p + 1 : i + 1].mean()
        mavp_ = Series(result, index=close.index)

    # Offset
    if offset != 0:
        mavp_ = mavp_.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        mavp_.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        if "fill_method" in kwargs:
            if kwargs["fill_method"] == "ffill":
                mavp_.ffill(inplace=True)
            elif kwargs["fill_method"] == "bfill":
                mavp_.bfill(inplace=True)

    # Name and Categorize it
    mavp_.name = f"MAVP_{minperiod}_{maxperiod}"
    mavp_.category = "overlap"

    return mavp_


mavp.__doc__ = """Moving Average with Variable Period (MAVP)

Moving Average with Variable Period computes a moving average where the
lookback period is different for each bar. The periods Series determines
the window size at each data point. When periods is None, a linearly
interpolated range from minperiod to maxperiod is used.

Sources:
    https://mrjbq7.github.io/ta-lib/func_groups/overlap_studies.html

Args:
    close (pd.Series): Close price series.
    periods (pd.Series): Variable period series (optional). Default: linearly spaced [minperiod, maxperiod]
    minperiod (int): Minimum allowed period. Default: 2
    maxperiod (int): Maximum allowed period. Default: 30
    mamode (int): MA type (TA-Lib convention: 0=SMA). Default: 0
    talib (bool): Use TA-Lib if installed. Default: True
    offset (int): Result offset. Default: 0

Returns:
    pd.Series: MAVP values.
"""
