# Directional Index (DX)
from typing import Any

from pandas import Series

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
from pandas_ta_classic.utils._core import _bool_param, _pos_float, _pos_int, _str_param


def dx(
    high: Series,
    low: Series,
    close: Series,
    length: int | None = None,
    scalar: float | None = None,
    mamode: str | None = None,
    talib: bool | None = None,
    drift: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Directional Index (DX)"""
    # Validate Arguments
    length = _pos_int(length, 14, "length")
    scalar = _pos_float(scalar, 100, "scalar")
    mamode = _str_param(mamode, "rma", "mamode")
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    drift = get_drift(drift)
    offset = get_offset(offset)
    mode_talib = _bool_param(talib, False, "talib")

    if high is None or low is None or close is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_talib:
        from talib import DX as TADX

        dx_ = TADX(high, low, close, length)
    else:
        up = high - high.shift(drift)
        dn = low.shift(drift) - low

        pos = ((up > dn) & (up > 0)) * up
        neg = ((dn > up) & (dn > 0)) * dn

        from pandas_ta_classic.utils import zero

        pos = pos.apply(zero)
        neg = neg.apply(zero)

        if mamode == "rma":
            # TA-Lib seeding (see utils/_wilder.py); DX is a ratio of the two
            # smoothed DMs, so true range cancels and is not needed. The seed
            # bar is not reported: TA-Lib's DX lookback is `length`.
            from pandas_ta_classic.utils._wilder import wilder_smooth

            dmp = wilder_smooth(pos, length)
            dmn = wilder_smooth(neg, length)
            seed = dmp.first_valid_index()
            if seed is not None:
                dmp[seed] = dmn[seed] = float("nan")
        else:
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
    talib (bool): Use TA-Lib if installed. Default: False
    drift (int): Drift period. Default: 1
    offset (int): Result offset. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: DX values.
"""
