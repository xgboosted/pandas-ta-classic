# -*- coding: utf-8 -*-
# Squeeze Pro (SQUEEZE_PRO)
from typing import Any, Optional
import numpy as np
from pandas import DataFrame, Series

npNaN = np.nan
from pandas_ta_classic.momentum import mom
from pandas_ta_classic.momentum.squeeze import (
    _pos_float,
    _pos_int,
    _squeeze_detailed,
    _squeeze_simplify_columns,
)
from pandas_ta_classic.overlap.ema import ema
from pandas_ta_classic.overlap.sma import sma
from pandas_ta_classic.trend import decreasing, increasing
from pandas_ta_classic.volatility import bbands, kc
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset
from pandas_ta_classic.utils import unsigned_differences, verify_series


def squeeze_pro(
    high: Series,
    low: Series,
    close: Series,
    bb_length: Optional[int] = None,
    bb_std: Optional[float] = None,
    kc_length: Optional[int] = None,
    kc_scalar_wide: Optional[float] = None,
    kc_scalar_normal: Optional[float] = None,
    kc_scalar_narrow: Optional[float] = None,
    mom_length: Optional[int] = None,
    mom_smooth: Optional[int] = None,
    use_tr: Optional[bool] = None,
    mamode: Optional[str] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: Squeeze Momentum (SQZ) PRO"""
    # Validate arguments
    bb_length = _pos_int(bb_length, 20)
    bb_std = _pos_float(bb_std, 2.0)
    kc_length = _pos_int(kc_length, 20)
    kc_scalar_wide = _pos_float(kc_scalar_wide, 2)
    kc_scalar_normal = _pos_float(kc_scalar_normal, 1.5)
    kc_scalar_narrow = _pos_float(kc_scalar_narrow, 1)
    mom_length = _pos_int(mom_length, 12)
    mom_smooth = _pos_int(mom_smooth, 6)

    _length = max(bb_length, kc_length, mom_length, mom_smooth)
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    close = verify_series(close, _length)
    offset = get_offset(offset)

    if not (kc_scalar_wide > kc_scalar_normal > kc_scalar_narrow):
        return None
    if high is None or low is None or close is None:
        return None

    use_tr = kwargs.setdefault("tr", True)
    asint = kwargs.pop("asint", True)
    detailed = kwargs.pop("detailed", False)
    mamode = mamode if isinstance(mamode, str) else "sma"

    # Calculate Result
    bbd = bbands(close, length=bb_length, std=bb_std, mamode=mamode)
    kch_wide = kc(
        high,
        low,
        close,
        length=kc_length,
        scalar=kc_scalar_wide,
        mamode=mamode,
        tr=use_tr,
    )
    kch_normal = kc(
        high,
        low,
        close,
        length=kc_length,
        scalar=kc_scalar_normal,
        mamode=mamode,
        tr=use_tr,
    )
    kch_narrow = kc(
        high,
        low,
        close,
        length=kc_length,
        scalar=kc_scalar_narrow,
        mamode=mamode,
        tr=use_tr,
    )
    if bbd is None or kch_wide is None or kch_normal is None or kch_narrow is None:
        return None

    # Simplify KC and BBAND column names for dynamic access
    bbd.columns = _squeeze_simplify_columns(bbd)
    kch_wide.columns = _squeeze_simplify_columns(kch_wide)
    kch_normal.columns = _squeeze_simplify_columns(kch_normal)
    kch_narrow.columns = _squeeze_simplify_columns(kch_narrow)

    momo = mom(close, length=mom_length)
    if mamode.lower() == "ema":
        squeeze = ema(momo, length=mom_smooth)
    else:  # "sma"
        squeeze = sma(momo, length=mom_smooth)

    # Classify Squeezes
    squeeze_on_wide = (bbd.l > kch_wide.l) & (bbd.u < kch_wide.u)
    squeeze_on_normal = (bbd.l > kch_normal.l) & (bbd.u < kch_normal.u)
    squeeze_on_narrow = (bbd.l > kch_narrow.l) & (bbd.u < kch_narrow.u)
    squeeze_off_wide = (bbd.l < kch_wide.l) & (bbd.u > kch_wide.u)
    no_squeeze = ~squeeze_on_wide & ~squeeze_off_wide

    # Offset
    (
        squeeze,
        squeeze_on_wide,
        squeeze_on_normal,
        squeeze_on_narrow,
        squeeze_off_wide,
        no_squeeze,
    ) = apply_offset(
        [
            squeeze,
            squeeze_on_wide,
            squeeze_on_normal,
            squeeze_on_narrow,
            squeeze_off_wide,
            no_squeeze,
        ],
        offset,
    )

    (
        squeeze,
        squeeze_on_wide,
        squeeze_on_normal,
        squeeze_on_narrow,
        squeeze_off_wide,
        no_squeeze,
    ) = apply_fill(
        [
            squeeze,
            squeeze_on_wide,
            squeeze_on_normal,
            squeeze_on_narrow,
            squeeze_off_wide,
            no_squeeze,
        ],
        **kwargs,
    )

    # Name and Categorize it
    _props = "" if use_tr else "hlr"
    _props += f"_{bb_length}_{bb_std}_{kc_length}_{kc_scalar_wide}_{kc_scalar_normal}_{kc_scalar_narrow}"
    squeeze.name = f"SQZPRO{_props}"

    data = {
        squeeze.name: squeeze,
        f"SQZPRO_ON_WIDE": squeeze_on_wide.astype(int) if asint else squeeze_on_wide,
        f"SQZPRO_ON_NORMAL": (
            squeeze_on_normal.astype(int) if asint else squeeze_on_normal
        ),
        f"SQZPRO_ON_NARROW": (
            squeeze_on_narrow.astype(int) if asint else squeeze_on_narrow
        ),
        f"SQZPRO_OFF": squeeze_off_wide.astype(int) if asint else squeeze_off_wide,
        f"SQZPRO_NO": no_squeeze.astype(int) if asint else no_squeeze,
    }
    df = DataFrame(data)
    df.name = squeeze.name
    df.category = squeeze.category = "momentum"

    # Detailed Squeeze Series
    if detailed:
        _squeeze_detailed(df, squeeze, kwargs)

    return df


squeeze_pro.__doc__ = """Squeeze PRO(SQZPRO)

This indicator is an extended version of "TTM Squeeze" from John Carter.
The default is based on John Carter's "TTM Squeeze" indicator, as discussed
in his book "Mastering the Trade" (chapter 11). The Squeeze indicator attempts
to capture the relationship between two studies: Bollinger Bands® and Keltner's
Channels. When the volatility increases, so does the distance between the bands,
conversely, when the volatility declines, the distance also decreases. It finds
sections of the Bollinger Bands® study which fall inside the Keltner's Channels.

Sources:
    https://usethinkscript.com/threads/john-carters-squeeze-pro-indicator-for-thinkorswim-free.4021/
    https://www.tradingview.com/script/TAAt6eRX-Squeeze-PRO-Indicator-Makit0/

Calculation:
    Default Inputs:
        bb_length=20, bb_std=2, kc_length=20, kc_scalar_wide=2,
        kc_scalar_normal=1.5, kc_scalar_narrow=1, mom_length=12,
        mom_smooth=6, tr=True,
    BB = Bollinger Bands
    KC = Keltner Channels
    MOM = Momentum
    SMA = Simple Moving Average
    EMA = Exponential Moving Average
    TR = True Range

    RANGE = TR(high, low, close) if using_tr else high - low
    BB_LOW, BB_MID, BB_HIGH = BB(close, bb_length, std=bb_std)
    KC_LOW_WIDE, KC_MID_WIDE, KC_HIGH_WIDE = KC(high, low, close, kc_length, kc_scalar_wide, TR)
    KC_LOW_NORMAL, KC_MID_NORMAL, KC_HIGH_NORMAL = KC(high, low, close, kc_length, kc_scalar_normal, TR)
    KC_LOW_NARROW, KC_MID_NARROW, KC_HIGH_NARROW = KC(high, low, close, kc_length, kc_scalar_narrow, TR)

    MOMO = MOM(close, mom_length)
    if mamode == "ema":
        SQZPRO = EMA(MOMO, mom_smooth)
    else:
        SQZPRO = EMA(momo, mom_smooth)

    SQZPRO_ON_WIDE  = (BB_LOW > KC_LOW_WIDE) and (BB_HIGH < KC_HIGH_WIDE)
    SQZPRO_ON_NORMAL  = (BB_LOW > KC_LOW_NORMAL) and (BB_HIGH < KC_HIGH_NORMAL)
    SQZPRO_ON_NARROW  = (BB_LOW > KC_LOW_NARROW) and (BB_HIGH < KC_HIGH_NARROW)
    SQZPRO_OFF_WIDE = (BB_LOW < KC_LOW_WIDE) and (BB_HIGH > KC_HIGH_WIDE)
    SQZPRO_NO = !SQZ_ON_WIDE and !SQZ_OFF_WIDE

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    bb_length (int): Bollinger Bands period. Default: 20
    bb_std (float): Bollinger Bands Std. Dev. Default: 2
    kc_length (int): Keltner Channel period. Default: 20
    kc_scalar_wide (float): Keltner Channel scalar for wider channel. Default: 2
    kc_scalar_normal (float): Keltner Channel scalar for normal channel. Default: 1.5
    kc_scalar_narrow (float): Keltner Channel scalar for narrow channel. Default: 1
    mom_length (int): Momentum Period. Default: 12
    mom_smooth (int): Smoothing Period of Momentum. Default: 6
    mamode (str): Only "ema" or "sma". Default: "sma"
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    tr (value, optional): Use True Range for Keltner Channels. Default: True
    asint (value, optional): Use integers instead of bool. Default: True
    mamode (value, optional): Which MA to use. Default: "sma"
    detailed (value, optional): Return additional variations of SQZ for
        visualization. Default: False
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: SQZPRO, SQZPRO_ON_WIDE, SQZPRO_ON_NORMAL, SQZPRO_ON_NARROW, SQZPRO_OFF_WIDE, SQZPRO_NO columns by default. More
        detailed columns if 'detailed' kwarg is True.
"""
