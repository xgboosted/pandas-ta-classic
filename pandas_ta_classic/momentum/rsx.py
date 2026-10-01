# Relative Strength Xtra (RSX)
from typing import Any

import numpy as np
from pandas import DataFrame, Series

from pandas_ta_classic.utils import get_offset, verify_series
from pandas_ta_classic.utils._core import _pos_int, nan_on_short_input, skip_leading_nan
from pandas_ta_classic.utils._njit import njit
from pandas_ta_classic.utils._signals import attach_signals


@njit(cache=True)
def _rsx_loop(c_arr, length, m):
    result = np.full(m, np.nan)
    result[length - 1] = 0.0
    vC = v1C = v4 = v8 = v10 = v14 = v18 = v20 = 0.0
    f0 = f8 = f10 = f18 = f20 = f28 = f30 = f38 = f40 = 0.0
    f48 = f50 = f58 = f60 = f68 = f70 = f78 = f80 = f88 = f90 = 0.0
    for i in range(length, m):
        if f90 == 0:
            f90 = 1.0
            f0 = 0.0
            f88 = length - 1.0 if length - 1.0 >= 5 else 5.0
            f8 = 100.0 * c_arr[i]
            f18 = 3.0 / (length + 2.0)
            f20 = 1.0 - f18
        else:
            f90 = f88 + 1 if f88 <= f90 else f90 + 1
            f10 = f8
            f8 = 100.0 * c_arr[i]
            v8 = f8 - f10
            f28 = f20 * f28 + f18 * v8
            f30 = f18 * f28 + f20 * f30
            vC = 1.5 * f28 - 0.5 * f30
            f38 = f20 * f38 + f18 * vC
            f40 = f18 * f38 + f20 * f40
            v10 = 1.5 * f38 - 0.5 * f40
            f48 = f20 * f48 + f18 * v10
            f50 = f18 * f48 + f20 * f50
            v14 = 1.5 * f48 - 0.5 * f50
            f58 = f20 * f58 + f18 * abs(v8)
            f60 = f18 * f58 + f20 * f60
            v18 = 1.5 * f58 - 0.5 * f60
            f68 = f20 * f68 + f18 * v18
            f70 = f18 * f68 + f20 * f70
            v1C = 1.5 * f68 - 0.5 * f70
            f78 = f20 * f78 + f18 * v1C
            f80 = f18 * f78 + f20 * f80
            v20 = 1.5 * f78 - 0.5 * f80
            if f88 >= f90 and f8 != f10:
                f0 = 1.0
            if f88 == f90 and f0 == 0.0:
                f90 = 0.0
        if f88 < f90 and v20 > 0.0000000001:
            v4 = (v14 / v20 + 1.0) * 50.0
            v4 = min(v4, 100.0)
            v4 = max(v4, 0.0)
        else:
            v4 = 50.0
        result[i] = v4
    return result


@nan_on_short_input
@skip_leading_nan("close", interior=True)
def rsx(
    close: Series,
    length: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | DataFrame | None:
    """Indicator: Relative Strength Xtra (inspired by Jurik RSX)"""
    # Validate arguments
    length = _pos_int(length, 14, "length")
    close = verify_series(close, length)
    offset = get_offset(offset)
    # A strategy-wide drift (df.ta.strategy(..., drift=N)) has no meaning here: the
    # parameter was removed in 0.9.0. Drop it rather than forward it to apply_fill.
    kwargs.pop("drift", None)

    if close is None:
        return None

    # Calculate Result
    m = close.size
    c_arr = close.to_numpy(dtype=float)
    rsx = Series(_rsx_loop(c_arr, length, m), index=close.index)

    # Name it here: the signal columns take their names from it, and `.name`
    # survives the shift while a custom attribute such as `.category` does not.
    rsx.name = f"RSX_{length}"

    # attach_signals owns the order of the remaining steps (signals off the
    # unoffset series, one shift, then the fill over every column).
    return attach_signals(rsx, category="momentum", offset=offset, kwargs=kwargs)


rsx.__doc__ = """Relative Strength Xtra (rsx)

The Relative Strength Xtra is based on the popular RSI indicator and inspired
by the work Jurik Research. The code implemented is based on published code
found at 'prorealcode.com'. This enhanced version of the rsi reduces noise and
provides a clearer, only slightly delayed insight on momentum and velocity of
price movements.

Sources:
    http://www.jurikres.com/catalog1/ms_rsx.htm
    https://www.prorealcode.com/prorealtime-indicators/jurik-rsx/

Calculation:
    Refer to the sources above for information as well as code example.

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 14
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    signal_indicators (bool): When True, threshold and comparison signal
        columns are appended and the result becomes a DataFrame instead of a
        Series. The options below are only read when it is True. Default: False
    xa (float): Upper threshold. Default: 80
    xb (float): Lower threshold. Default: 20
    cross_values (bool): When True, the xa/xb columns mark the bars that cross
        the threshold instead of flagging every bar on one side of it.
        Default: False
    xserie (pd.Series): Comparison series; used for both xserie_a and xserie_b
        unless one of them is given. Default: None
    xserie_a (pd.Series): Comparison series for the "above" column.
        Default: None
    xserie_b (pd.Series): Comparison series for the "below" column.
        Default: None
    cross_series (bool): When True, the xserie columns mark crossings instead
        of flagging every bar on one side of the series. Default: True
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated, or a pd.DataFrame with the indicator and
        its signal columns when signal_indicators is True.
"""
