# -*- coding: utf-8 -*-
# Parabolic SAR Extended (SAREXT)
from typing import Any, Optional

import numpy as np
from pandas import Series

from pandas_ta_classic import Imports
from pandas_ta_classic.utils import get_offset, verify_series, zero
from pandas_ta_classic.utils._njit import njit

npNaN = np.nan


@njit(cache=True)
def _sarext_loop(
    h_arr,
    l_arr,
    m,
    falling,
    sar,
    ep,
    af0_long,
    af_long,
    max_af_long,
    af0_short,
    af_short,
    max_af_short,
    offset_on_reverse,
):
    long_arr = np.full(m, np.nan)
    short_arr = np.full(m, np.nan)
    af_arr = np.full(m, np.nan)
    af = af0_short if falling else af0_long
    af_arr[0] = af

    for row in range(1, m):
        h_ = h_arr[row]
        l_ = l_arr[row]

        if falling:
            _sar = sar + af * (ep - sar)
            reverse = h_ > _sar
            if l_ < ep:
                ep = l_
                af = min(af + af_short, max_af_short)
            _sar = max(h_arr[row - 1], h_arr[max(0, row - 2)], _sar)
        else:
            _sar = sar + af * (ep - sar)
            reverse = l_ < _sar
            if h_ > ep:
                ep = h_
                af = min(af + af_long, max_af_long)
            _sar = min(l_arr[row - 1], l_arr[max(0, row - 2)], _sar)

        if reverse:
            if offset_on_reverse != 0.0:
                _sar = (
                    _sar + offset_on_reverse * _sar
                    if falling
                    else _sar - offset_on_reverse * _sar
                )
            falling = not falling
            if falling:
                ep = l_
                af = af0_short
            else:
                ep = h_
                af = af0_long

        sar = _sar
        if falling:
            short_arr[row] = sar
        else:
            long_arr[row] = sar
        af_arr[row] = af

    return long_arr, short_arr, af_arr


def sarext(
    high: Series,
    low: Series,
    startvalue: Optional[float] = None,
    offsetonreverse: Optional[float] = None,
    accelerationinitlong: Optional[float] = None,
    accelerationlong: Optional[float] = None,
    accelerationmaxlong: Optional[float] = None,
    accelerationinitshort: Optional[float] = None,
    accelerationshort: Optional[float] = None,
    accelerationmaxshort: Optional[float] = None,
    talib: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Parabolic SAR Extended (SAREXT)"""
    # Validate Arguments
    high = verify_series(high)
    low = verify_series(low)
    startvalue = float(startvalue) if startvalue is not None else 0.0
    offsetonreverse = float(offsetonreverse) if offsetonreverse is not None else 0.0
    af0_long = (
        float(accelerationinitlong)
        if accelerationinitlong and accelerationinitlong > 0
        else 0.02
    )
    af_long = (
        float(accelerationlong) if accelerationlong and accelerationlong > 0 else 0.02
    )
    max_af_long = (
        float(accelerationmaxlong)
        if accelerationmaxlong and accelerationmaxlong > 0
        else 0.2
    )
    af0_short = (
        float(accelerationinitshort)
        if accelerationinitshort and accelerationinitshort > 0
        else 0.02
    )
    af_short = (
        float(accelerationshort)
        if accelerationshort and accelerationshort > 0
        else 0.02
    )
    max_af_short = (
        float(accelerationmaxshort)
        if accelerationmaxshort and accelerationmaxshort > 0
        else 0.2
    )
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    if high is None or low is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from talib import SAREXT as TASAREXT

        sarext_ = TASAREXT(
            high,
            low,
            startvalue=startvalue,
            offsetonreverse=offsetonreverse,
            accelerationinitlong=af0_long,
            accelerationlong=af_long,
            accelerationmaxlong=max_af_long,
            accelerationinitshort=af0_short,
            accelerationshort=af_short,
            accelerationmaxshort=max_af_short,
        )
    else:
        # Native implementation
        def _falling(high: Series, low: Series, drift: int = 1) -> bool:
            up = high - high.shift(drift)
            dn = low.shift(drift) - low
            _dmn = (((dn > up) & (dn > 0)) * dn).apply(zero).iloc[-1]
            return _dmn > 0

        falling = _falling(high.iloc[:2], low.iloc[:2]) if len(high) > 1 else False

        if startvalue != 0.0:
            sar = startvalue
        elif falling:
            sar = high.iloc[0]
        else:
            sar = low.iloc[0]

        ep = (
            low.iloc[1]
            if falling and len(low) > 1
            else (high.iloc[1] if len(high) > 1 else high.iloc[0])
        )

        m = high.shape[0]
        h_arr = high.to_numpy(dtype=float)
        l_arr = low.to_numpy(dtype=float)

        long_arr, short_arr, af_arr = _sarext_loop(
            h_arr,
            l_arr,
            m,
            falling,
            sar,
            ep,
            af0_long,
            af_long,
            max_af_long,
            af0_short,
            af_short,
            max_af_short,
            offsetonreverse,
        )

        # SAREXT returns signed values: positive = long SAR, negative = short SAR
        from pandas import Series as _Series

        result = np.where(
            ~np.isnan(long_arr),
            long_arr,
            np.where(~np.isnan(short_arr), -short_arr, np.nan),
        )
        sarext_ = _Series(result, index=high.index)

    # Offset
    if offset != 0:
        sarext_ = sarext_.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        sarext_.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        if "fill_method" in kwargs:
            if kwargs["fill_method"] == "ffill":
                sarext_.ffill(inplace=True)
            elif kwargs["fill_method"] == "bfill":
                sarext_.bfill(inplace=True)

    # Name and Categorize it
    sarext_.name = "SAREXT"
    sarext_.category = "trend"

    return sarext_


sarext.__doc__ = """Parabolic SAR Extended (SAREXT)

The Parabolic SAR Extended is an enhanced version of the Parabolic SAR that
allows separate acceleration factor settings for long and short positions,
plus an optional offset applied when a reversal occurs.

Sources:
    https://mrjbq7.github.io/ta-lib/func_groups/overlap_studies.html

Args:
    high (pd.Series): High price series.
    low (pd.Series): Low price series.
    startvalue (float): Starting SAR value (0 = use first high/low). Default: 0
    offsetonreverse (float): Fractional offset added to SAR on reversal. Default: 0
    accelerationinitlong (float): Initial AF for long positions. Default: 0.02
    accelerationlong (float): AF increment for long positions. Default: 0.02
    accelerationmaxlong (float): Maximum AF for long positions. Default: 0.2
    accelerationinitshort (float): Initial AF for short positions. Default: 0.02
    accelerationshort (float): AF increment for short positions. Default: 0.02
    accelerationmaxshort (float): Maximum AF for short positions. Default: 0.2
    talib (bool): Use TA-Lib if installed. Default: True
    offset (int): Result offset. Default: 0

Returns:
    pd.Series: SAREXT values (positive = long SAR, negative = short SAR).
"""
