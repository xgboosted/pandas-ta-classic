# -*- coding: utf-8 -*-
# Extended Parabolic SAR (SAREXT)
from typing import Any, Optional

import numpy as np
from pandas import Series

from pandas_ta_classic import Imports
from pandas_ta_classic.utils import (
    _get_tal_mode,
    _finalize,
    get_offset,
    verify_series,
    zero,
)


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
    """Indicator: Extended Parabolic SAR (SAREXT)"""
    # Validate Arguments
    high = verify_series(high)
    low = verify_series(low)
    offset = get_offset(offset)
    mode_tal = _get_tal_mode(talib)

    startvalue = float(startvalue) if startvalue is not None else 0.0
    offsetonreverse = float(offsetonreverse) if offsetonreverse is not None else 0.0
    accelerationinitlong = (
        float(accelerationinitlong)
        if accelerationinitlong and accelerationinitlong > 0
        else 0.02
    )
    accelerationlong = (
        float(accelerationlong) if accelerationlong and accelerationlong > 0 else 0.02
    )
    accelerationmaxlong = (
        float(accelerationmaxlong)
        if accelerationmaxlong and accelerationmaxlong > 0
        else 0.2
    )
    accelerationinitshort = (
        float(accelerationinitshort)
        if accelerationinitshort and accelerationinitshort > 0
        else 0.02
    )
    accelerationshort = (
        float(accelerationshort)
        if accelerationshort and accelerationshort > 0
        else 0.02
    )
    accelerationmaxshort = (
        float(accelerationmaxshort)
        if accelerationmaxshort and accelerationmaxshort > 0
        else 0.2
    )

    if high is None or low is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from talib import SAREXT as taSAREXT

        result = Series(
            taSAREXT(
                high,
                low,
                startvalue=startvalue,
                offsetonreverse=offsetonreverse,
                accelerationinitlong=accelerationinitlong,
                accelerationlong=accelerationlong,
                accelerationmaxlong=accelerationmaxlong,
                accelerationinitshort=accelerationinitshort,
                accelerationshort=accelerationshort,
                accelerationmaxshort=accelerationmaxshort,
            ),
            index=high.index,
        )
    else:
        from pandas_ta_classic.utils._numba import _sarext_loop

        h_arr = high.to_numpy(dtype=float)
        l_arr = low.to_numpy(dtype=float)
        m = h_arr.shape[0]

        if m < 2:
            return None

        # Determine initial direction and SAR/EP (TA-Lib compatible).
        # TA-Lib: MINUS_DM on bar 1 determines direction when startvalue=0.
        # EP is always from bar 1; SAR from bar 0 (or startvalue).
        if startvalue == 0.0:
            # Check MINUS_DM(1) on bars 0-1
            def _falling(high_s: Series, low_s: Series, drift: int = 1) -> bool:
                up = high_s - high_s.shift(drift)
                dn = low_s.shift(drift) - low_s
                _dmn = (((dn > up) & (dn > 0)) * dn).apply(zero).iloc[-1]
                return _dmn > 0

            is_long = not _falling(high.iloc[:2], low.iloc[:2])
            if is_long:
                sar = l_arr[0]
                ep = h_arr[1]
            else:
                sar = h_arr[0]
                ep = l_arr[1]
        elif startvalue > 0.0:
            is_long = True
            sar = startvalue
            ep = h_arr[1]
        else:
            is_long = False
            sar = abs(startvalue)
            ep = l_arr[1]

        result_arr = _sarext_loop(
            h_arr,
            l_arr,
            m,
            is_long,
            sar,
            ep,
            accelerationinitlong,
            accelerationlong,
            accelerationmaxlong,
            accelerationinitshort,
            accelerationshort,
            accelerationmaxshort,
            offsetonreverse,
        )
        result = Series(result_arr, index=high.index)

    _params = f"_{accelerationinitlong}_{accelerationmaxlong}"
    return _finalize(result, offset, f"SAREXT{_params}", "trend", **kwargs)


sarext.__doc__ = """Extended Parabolic SAR (SAREXT)

An extended version of the Parabolic SAR that allows separate acceleration
factor parameters for long and short positions.  The output is a single
series where positive values represent the long SAR and negative values
represent the short SAR.

Sources:
    TA Lib

Calculation:
    Default Inputs:
        startvalue=0, offsetonreverse=0,
        accelerationinitlong=0.02, accelerationlong=0.02, accelerationmaxlong=0.2,
        accelerationinitshort=0.02, accelerationshort=0.02, accelerationmaxshort=0.2

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    startvalue (float): Start value. Default: 0
    offsetonreverse (float): Offset added on reversal. Default: 0
    accelerationinitlong (float): Initial AF for long. Default: 0.02
    accelerationlong (float): AF step for long. Default: 0.02
    accelerationmaxlong (float): Max AF for long. Default: 0.2
    accelerationinitshort (float): Initial AF for short. Default: 0.02
    accelerationshort (float): AF step for short. Default: 0.02
    accelerationmaxshort (float): Max AF for short. Default: 0.2
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated (positive=long, negative=short).
"""
