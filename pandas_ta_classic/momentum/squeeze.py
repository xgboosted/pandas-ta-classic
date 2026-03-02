# Squeeze (SQUEEZE)
from typing import Any, Optional
import numpy as np
from pandas import DataFrame, Series

npNaN = np.nan
from pandas_ta_classic.momentum import mom
from pandas_ta_classic.overlap.ema import ema
from pandas_ta_classic.overlap.linreg import linreg
from pandas_ta_classic.overlap.sma import sma
from pandas_ta_classic.trend import decreasing, increasing
from pandas_ta_classic.volatility import bbands, kc
from pandas_ta_classic.utils import (
    _build_dataframe,
    apply_offset,
    get_offset,
    unsigned_differences,
    verify_series,
)


def squeeze(
    high: Series,
    low: Series,
    close: Series,
    bb_length: Optional[int] = None,
    bb_std: Optional[float] = None,
    kc_length: Optional[int] = None,
    kc_scalar: Optional[float] = None,
    mom_length: Optional[int] = None,
    mom_smooth: Optional[int] = None,
    use_tr: Optional[bool] = None,
    mamode: Optional[str] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: Squeeze Momentum (SQZ)"""
    # Validate arguments
    bb_length = int(bb_length) if bb_length and bb_length > 0 else 20
    bb_std = float(bb_std) if bb_std and bb_std > 0 else 2.0
    kc_length = int(kc_length) if kc_length and kc_length > 0 else 20
    kc_scalar = float(kc_scalar) if kc_scalar and kc_scalar > 0 else 1.5
    mom_length = int(mom_length) if mom_length and mom_length > 0 else 12
    mom_smooth = int(mom_smooth) if mom_smooth and mom_smooth > 0 else 6
    _length = max(bb_length, kc_length, mom_length, mom_smooth)
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    close = verify_series(close, _length)
    offset = get_offset(offset)

    if high is None or low is None or close is None:
        return None

    use_tr = kwargs.setdefault("tr", True)
    asint = kwargs.pop("asint", True)
    detailed = kwargs.pop("detailed", False)
    lazybear = kwargs.pop("lazybear", False)
    mamode = mamode if isinstance(mamode, str) else "sma"

    def simplify_columns(df: DataFrame, n: int = 3) -> list:
        df.columns = df.columns.str.lower()
        return [c.split("_")[0][n - 1 : n] for c in df.columns]

    # Calculate Result
    bbd = bbands(close, length=bb_length, std=bb_std, mamode=mamode)
    kch = kc(
        high, low, close, length=kc_length, scalar=kc_scalar, mamode=mamode, tr=use_tr
    )

    # Simplify KC and BBAND column names for dynamic access
    bbd.columns = simplify_columns(bbd)
    kch.columns = simplify_columns(kch)

    if lazybear:
        highest_high = high.rolling(kc_length).max()
        lowest_low = low.rolling(kc_length).min()
        avg_ = 0.25 * (highest_high + lowest_low) + 0.5 * kch.b

        squeeze = linreg(close - avg_, length=kc_length)

    else:
        momo = mom(close, length=mom_length)
        if mamode.lower() == "ema":
            squeeze = ema(momo, length=mom_smooth)
        else:  # "sma"
            squeeze = sma(momo, length=mom_smooth)

    # Classify Squeezes
    squeeze_on = (bbd.l > kch.l) & (bbd.u < kch.u)
    squeeze_off = (bbd.l < kch.l) & (bbd.u > kch.u)
    no_squeeze = ~squeeze_on & ~squeeze_off

    # Convert bool flags to int before offset so NaN-safe shift works
    if asint:
        squeeze_on = squeeze_on.astype(int)
        squeeze_off = squeeze_off.astype(int)
        no_squeeze = no_squeeze.astype(int)

    # Offset + Name + Category + DataFrame
    _props = "" if use_tr else "hlr"
    _props += f"_{bb_length}_{bb_std}_{kc_length}_{kc_scalar}"
    _props += "_LB" if lazybear else ""

    df = _build_dataframe(
        {
            f"SQZ{_props}": squeeze,
            "SQZ_ON": squeeze_on,
            "SQZ_OFF": squeeze_off,
            "SQZ_NO": no_squeeze,
        },
        f"SQZ{_props}",
        "momentum",
        offset,
        **kwargs,
    )

    # Detailed Squeeze Series
    if detailed:
        sqz = df.iloc[:, 0]
        pos_squeeze = sqz[sqz >= 0]
        neg_squeeze = sqz[sqz < 0]

        pos_inc, pos_dec = unsigned_differences(pos_squeeze, asint=True)
        neg_inc, neg_dec = unsigned_differences(neg_squeeze, asint=True)

        pos_inc *= sqz
        pos_dec *= sqz
        neg_dec *= sqz
        neg_inc *= sqz

        pos_inc = pos_inc.replace(0, npNaN)
        pos_dec = pos_dec.replace(0, npNaN)
        neg_dec = neg_dec.replace(0, npNaN)
        neg_inc = neg_inc.replace(0, npNaN)

        sqz_inc = sqz * increasing(sqz)
        sqz_dec = sqz * decreasing(sqz)
        sqz_inc = sqz_inc.replace(0, npNaN)
        sqz_dec = sqz_dec.replace(0, npNaN)

        sqz_inc = apply_offset(sqz_inc, 0, **kwargs)
        sqz_dec = apply_offset(sqz_dec, 0, **kwargs)
        pos_inc = apply_offset(pos_inc, 0, **kwargs)
        pos_dec = apply_offset(pos_dec, 0, **kwargs)
        neg_dec = apply_offset(neg_dec, 0, **kwargs)
        neg_inc = apply_offset(neg_inc, 0, **kwargs)

        df["SQZ_INC"] = sqz_inc
        df["SQZ_DEC"] = sqz_dec
        df["SQZ_PINC"] = pos_inc
        df["SQZ_PDEC"] = pos_dec
        df["SQZ_NDEC"] = neg_dec
        df["SQZ_NINC"] = neg_inc

    return df


squeeze.__doc__ = """Squeeze (SQZ)

The default is based on John Carter's "TTM Squeeze" indicator, as discussed
in his book "Mastering the Trade" (chapter 11). The Squeeze indicator attempts
to capture the relationship between two studies: Bollinger Bands® and Keltner's
Channels. When the volatility increases, so does the distance between the bands,
conversely, when the volatility declines, the distance also decreases. It finds
sections of the Bollinger Bands® study which fall inside the Keltner's Channels.

Sources:
    https://tradestation.tradingappstore.com/products/TTMSqueeze
    https://www.tradingview.com/scripts/lazybear/
    https://tlc.thinkorswim.com/center/reference/Tech-Indicators/studies-library/T-U/TTM-Squeeze

Calculation:
    Default Inputs:
        bb_length=20, bb_std=2, kc_length=20, kc_scalar=1.5, mom_length=12,
        mom_smooth=12, tr=True, lazybear=False,
    BB = Bollinger Bands
    KC = Keltner Channels
    MOM = Momentum
    SMA = Simple Moving Average
    EMA = Exponential Moving Average
    TR = True Range

    RANGE = TR(high, low, close) if using_tr else high - low
    BB_LOW, BB_MID, BB_HIGH = BB(close, bb_length, std=bb_std)
    KC_LOW, KC_MID, KC_HIGH = KC(high, low, close, kc_length, kc_scalar, TR)

    if lazybear:
        HH = high.rolling(kc_length).max()
        LL = low.rolling(kc_length).min()
        AVG  = 0.25 * (HH + LL) + 0.5 * KC_MID
        SQZ = linreg(close - AVG, kc_length)
    else:
        MOMO = MOM(close, mom_length)
        if mamode == "ema":
            SQZ = EMA(MOMO, mom_smooth)
        else:
            SQZ = EMA(momo, mom_smooth)

    SQZ_ON  = (BB_LOW > KC_LOW) and (BB_HIGH < KC_HIGH)
    SQZ_OFF = (BB_LOW < KC_LOW) and (BB_HIGH > KC_HIGH)
    NO_SQZ = !SQZ_ON and !SQZ_OFF

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    bb_length (int): Bollinger Bands period. Default: 20
    bb_std (float): Bollinger Bands Std. Dev. Default: 2
    kc_length (int): Keltner Channel period. Default: 20
    kc_scalar (float): Keltner Channel scalar. Default: 1.5
    mom_length (int): Momentum Period. Default: 12
    mom_smooth (int): Smoothing Period of Momentum. Default: 6
    mamode (str): Only "ema" or "sma". Default: "sma"
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    tr (value, optional): Use True Range for Keltner Channels. Default: True
    asint (value, optional): Use integers instead of bool. Default: True
    mamode (value, optional): Which MA to use. Default: "sma"
    lazybear (value, optional): Use LazyBear's TradingView implementation.
        Default: False
    detailed (value, optional): Return additional variations of SQZ for
        visualization. Default: False
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: SQZ, SQZ_ON, SQZ_OFF, NO_SQZ columns by default. More
        detailed columns if 'detailed' kwarg is True.
"""
