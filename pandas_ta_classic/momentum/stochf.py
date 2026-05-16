# -*- coding: utf-8 -*-
# Stochastic Fast (STOCHF)
from typing import Any, Optional

import numpy as np
from pandas import DataFrame, Series

from pandas_ta_classic import Imports
from pandas_ta_classic.overlap.ma import ma
from pandas_ta_classic.utils import (
    apply_fill,
    apply_offset,
    get_offset,
    verify_series,
)


def _stochf_native(high, low, close, fastk, fastd, mamode):
    """Compute Stochastic Fast natively (no TA-Lib).

    Implements TA-Lib STOCHF exactly: scalar loop for k (avoids vectorized
    FP rounding differences), SMA for d when mamode="sma", and masks the
    first fastk-1+fastd-1 bars of k to NaN to match TA-Lib's output lookback.

    Args:
        high (Series): High price series.
        low (Series): Low price series.
        close (Series): Close price series.
        fastk (int): Fast %K look-back period.
        fastd (int): Fast %D smoothing period.
        mamode (str): MA type for %D smoothing.

    Returns:
        tuple[Series, Series] | None: ``(fastk_, fastd_)`` or *None* when
        the %D MA computation fails.
    """
    h_arr = high.to_numpy(dtype=float)
    l_arr = low.to_numpy(dtype=float)
    c_arr = close.to_numpy(dtype=float)
    m = len(c_arr)

    # Scalar loop matches TA-Lib's C doubles arithmetic exactly.
    k_arr = np.full(m, np.nan)
    for i in range(fastk - 1, m):
        lo = l_arr[i - fastk + 1 : i + 1].min()
        hi = h_arr[i - fastk + 1 : i + 1].max()
        diff = hi - lo
        k_arr[i] = 0.0 if diff == 0.0 else 100.0 * (c_arr[i] - lo) / diff

    # Compute d using raw k values (before masking k output).
    d_arr = np.full(m, np.nan)
    total_lookback = fastk - 1 + fastd - 1
    if mamode.lower() == "sma":
        for i in range(total_lookback, m):
            d_arr[i] = k_arr[i - fastd + 1 : i + 1].mean()
    else:
        # Fall back to pandas ma for non-SMA modes.
        k_tmp = Series(k_arr, index=close.index)
        first_valid = k_tmp.first_valid_index()
        if first_valid is None:
            return k_tmp, k_tmp.copy()
        d_tmp = ma(mamode, k_tmp.loc[first_valid:], length=fastd)
        if d_tmp is None:
            return None
        d_arr = d_tmp.reindex(close.index).to_numpy(dtype=float)

    # TA-Lib masks the first fastk-1+fastd-1 bars of k (used internally
    # for d but not shown in output).
    k_arr[:total_lookback] = np.nan

    fastk_ = Series(k_arr, index=close.index)
    fastd_ = Series(d_arr, index=close.index)
    return fastk_, fastd_


def stochf(
    high: Series,
    low: Series,
    close: Series,
    fastk: Optional[int] = None,
    fastd: Optional[int] = None,
    mamode: Optional[str] = None,
    talib: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: Stochastic Fast (STOCHF)"""
    # Validate Arguments
    fastk = fastk if fastk and fastk > 0 else 5
    fastd = fastd if fastd and fastd > 0 else 3
    _length = max(fastk, fastd)
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    close = verify_series(close, _length)
    offset = get_offset(offset)
    mamode = mamode if isinstance(mamode, str) else "sma"
    mode_talib = bool(talib) if isinstance(talib, bool) else False

    if high is None or low is None or close is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_talib:
        from talib import STOCHF as TASTOCHF

        fastk_, fastd_ = TASTOCHF(high, low, close, fastk, fastd)
    else:
        result = _stochf_native(high, low, close, fastk, fastd, mamode)
        if result is None:
            return None
        fastk_, fastd_ = result

    # Offset
    fastk_, fastd_ = apply_offset([fastk_, fastd_], offset)

    fastk_, fastd_ = apply_fill([fastk_, fastd_], **kwargs)

    # Name and Categorize it
    _name = "STOCHF"
    _params = f"_{fastk}_{fastd}"
    fastk_.name = f"{_name}k{_params}"
    fastd_.name = f"{_name}d{_params}"
    fastk_.category = fastd_.category = "momentum"

    data = {fastk_.name: fastk_, fastd_.name: fastd_}
    df = DataFrame(data)
    df.name = f"{_name}{_params}"
    df.category = fastk_.category

    return df


stochf.__doc__ = """Stochastic Fast (STOCHF)

The Stochastic Fast oscillator is a faster variant of the classic Stochastic
oscillator. It uses a shorter smoothing period for %D, making it more
responsive to price changes.

%FastK = 100 * (Close - Lowest Low[n]) / (Highest High[n] - Lowest Low[n])
%FastD = MA(%FastK, fastd_period)

Sources:
    https://www.investopedia.com/terms/s/stochasticoscillator.asp

Args:
    high (pd.Series): High price series.
    low (pd.Series): Low price series.
    close (pd.Series): Close price series.
    fastk (int): Fast %K period. Default: 5
    fastd (int): Fast %D smoothing period. Default: 3
    mamode (str): MA type for %D. Default: 'sma'
    talib (bool): Use TA-Lib if installed. Default: False
    offset (int): Result offset. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: FastK and FastD columns.
"""
