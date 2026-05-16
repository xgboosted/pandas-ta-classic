# -*- coding: utf-8 -*-
# Directional Index (DX)
from typing import Any, Optional

import numpy as np
from pandas import DataFrame, Series

from pandas_ta_classic import Imports
from pandas_ta_classic.overlap.ma import ma
from pandas_ta_classic.utils import (
    apply_fill,
    apply_offset,
    get_drift,
    get_offset,
    non_zero_range,
    verify_series,
)


def dx(
    high: Series,
    low: Series,
    close: Series,
    length: Optional[int] = None,
    scalar: Optional[float] = None,
    mamode: Optional[str] = None,
    talib: Optional[bool] = None,
    drift: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Directional Index (DX)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 14
    scalar = float(scalar) if scalar and scalar > 0 else 100
    mamode = mamode.lower() if isinstance(mamode, str) else "rma"
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    drift = get_drift(drift)
    offset = get_offset(offset)
    mode_talib = bool(talib) if isinstance(talib, bool) else True

    if high is None or low is None or close is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_talib:
        from talib import DX as TADX

        dx_ = TADX(high, low, close, length)
    elif mamode == "rma":
        # TA-Lib-compatible Wilder's smoothing for DX.
        # TA-Lib seeds with bars 1..length-1, then applies the first Wilder
        # update at bar `length`. This matches TA-Lib's internal lookback and
        # produces an exact numerical match when compared to tal.DX().
        h_arr = high.values.astype(float)
        l_arr = low.values.astype(float)
        c_arr = close.values.astype(float)
        n = len(h_arr)

        dm_pos = np.zeros(n)
        dm_neg = np.zeros(n)
        tr_raw = np.zeros(n)
        for idx in range(1, n):
            up = h_arr[idx] - h_arr[idx - 1]
            dn = l_arr[idx - 1] - l_arr[idx]
            dm_pos[idx] = up if (up > dn and up > 0) else 0.0
            dm_neg[idx] = dn if (dn > up and dn > 0) else 0.0
            tr_raw[idx] = max(
                h_arr[idx] - l_arr[idx],
                abs(h_arr[idx] - c_arr[idx - 1]),
                abs(l_arr[idx] - c_arr[idx - 1]),
            )

        if n <= length:
            return None

        # Seed: sum of bars 1..length-1, then first Wilder update at bar `length`
        tr14 = tr_raw[1:length].sum()
        dmpos14 = dm_pos[1:length].sum()
        dmneg14 = dm_neg[1:length].sum()
        tr14 = tr14 - tr14 / length + tr_raw[length]
        dmpos14 = dmpos14 - dmpos14 / length + dm_pos[length]
        dmneg14 = dmneg14 - dmneg14 / length + dm_neg[length]

        dx_raw = np.full(n, np.nan)
        denom = dmpos14 + dmneg14
        dx_raw[length] = scalar * abs(dmpos14 - dmneg14) / denom if denom != 0 else 0.0

        _tr14, _dmpos14, _dmneg14 = tr14, dmpos14, dmneg14
        for idx in range(length + 1, n):
            _tr14 = _tr14 - _tr14 / length + tr_raw[idx]
            _dmpos14 = _dmpos14 - _dmpos14 / length + dm_pos[idx]
            _dmneg14 = _dmneg14 - _dmneg14 / length + dm_neg[idx]
            denom = _dmpos14 + _dmneg14
            dx_raw[idx] = (
                scalar * abs(_dmpos14 - _dmneg14) / denom if denom != 0 else 0.0
            )

        dx_ = Series(dx_raw, index=close.index, dtype=float)
    else:
        up = high - high.shift(drift)
        dn = low.shift(drift) - low

        pos = ((up > dn) & (up > 0)) * up
        neg = ((dn > up) & (dn > 0)) * dn

        from pandas_ta_classic.utils import zero

        pos = pos.apply(zero)
        neg = neg.apply(zero)

        dmp = ma(mamode, pos, length=length)
        dmn = ma(mamode, neg, length=length)

        if dmp is None or dmn is None:
            return None

        dx_ = scalar * (dmp - dmn).abs() / non_zero_range(dmp, -dmn)

    # Offset
    dx_ = apply_offset(dx_, offset)

    dx_ = apply_fill(dx_, **kwargs)

    # Name and Categorize it
    dx_.name = f"DX_{length}"
    dx_.category = "trend"

    return dx_


dx.__doc__ = """Directional Index (DX)

The Directional Index (DX) is an intermediate step in calculating the Average
Directional Index (ADX). It measures the strength of trend direction by
comparing positive and negative directional movements.

Sources:
    https://www.investopedia.com/terms/d/dmi.asp

Args:
    high (pd.Series): High price series.
    low (pd.Series): Low price series.
    close (pd.Series): Close price series.
    length (int): The period. Default: 14
    scalar (float): Scalar multiplier. Default: 100
    mamode (str): Smoothing mode. Default: 'rma'
    talib (bool): Use TA-Lib if installed. Default: True
    drift (int): Drift period. Default: 1
    offset (int): Result offset. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: DX values.
"""
