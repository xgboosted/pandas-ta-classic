# Squeeze (SQUEEZE)
from typing import Any, Optional
import numpy as np
from pandas import DataFrame, Series


from pandas_ta_classic.momentum.mom import mom
from pandas_ta_classic.overlap.ema import ema
from pandas_ta_classic.overlap.linreg import linreg
from pandas_ta_classic.overlap.sma import sma
from pandas_ta_classic.trend.decreasing import decreasing
from pandas_ta_classic.trend.increasing import increasing
from pandas_ta_classic.volatility.bbands import bbands
from pandas_ta_classic.volatility.kc import kc
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset
from pandas_ta_classic.utils import unsigned_differences, verify_series
from pandas_ta_classic.utils._core import _pos_float, _pos_int


def _squeeze_simplify_columns(df, n=3):
    """Return shortened column name list for a bbands / kc DataFrame.

    Lowercases all column names then extracts the character at position
    ``n - 1`` of the first ``_``-delimited segment (e.g. ``"bbl_20_2.0"``
    → ``"l"``).

    Args:
        df (DataFrame): bbands or kc result frame.
        n (int): 1-based position of the character to keep. Default: 3.

    Returns:
        list[str]: Shortened column name strings (one per column).
    """
    df.columns = df.columns.str.lower()
    return [c.split("_")[0][n - 1 : n] for c in df.columns]


def _squeeze_momentum(close, high, low, kch_b, mom_length, mom_smooth, kc_length, mamode, lazybear):
    """Compute the squeeze momentum series (lazybear or standard path)."""
    if lazybear:
        highest_high = high.rolling(kc_length).max()
        lowest_low = low.rolling(kc_length).min()
        avg_ = 0.25 * (highest_high + lowest_low) + 0.5 * kch_b
        return linreg(close - avg_, length=kc_length)
    momo = mom(close, length=mom_length)
    if mamode.lower() == "ema":
        return ema(momo, length=mom_smooth)
    return sma(momo, length=mom_smooth)


def _squeeze_detailed(df, squeeze_s, kwargs, prefix="SQZ_"):
    """Append detailed signed-momentum columns to *df* in place.

    Breaks the squeeze series into positive / negative sub-series, computes
    increasing / decreasing variants, and attaches them as additional columns.

    Args:
        df (DataFrame): The base squeeze result frame (modified in place).
        squeeze_s (Series): The squeeze momentum series.
        kwargs (dict): Forwarded to :func:`apply_fill`.
        prefix (str): Column name prefix. Default ``"SQZ_"``.
    """
    pos_squeeze = squeeze_s[squeeze_s >= 0]
    neg_squeeze = squeeze_s[squeeze_s < 0]

    pos_inc, pos_dec = unsigned_differences(pos_squeeze, asint=True)
    neg_inc, neg_dec = unsigned_differences(neg_squeeze, asint=True)

    pos_inc *= squeeze_s
    pos_dec *= squeeze_s
    neg_dec *= squeeze_s
    neg_inc *= squeeze_s

    pos_inc.replace(0, np.nan, inplace=True)
    pos_dec.replace(0, np.nan, inplace=True)
    neg_dec.replace(0, np.nan, inplace=True)
    neg_inc.replace(0, np.nan, inplace=True)

    sqz_inc = squeeze_s * increasing(squeeze_s)
    sqz_dec = squeeze_s * decreasing(squeeze_s)
    sqz_inc.replace(0, np.nan, inplace=True)
    sqz_dec.replace(0, np.nan, inplace=True)

    sqz_inc, sqz_dec, pos_inc, pos_dec, neg_dec, neg_inc = apply_fill([sqz_inc, sqz_dec, pos_inc, pos_dec, neg_dec, neg_inc], **kwargs)

    df[f"{prefix}INC"] = sqz_inc
    df[f"{prefix}DEC"] = sqz_dec
    df[f"{prefix}PINC"] = pos_inc
    df[f"{prefix}PDEC"] = pos_dec
    df[f"{prefix}NDEC"] = neg_dec
    df[f"{prefix}NINC"] = neg_inc


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
    bb_length = _pos_int(bb_length, 20)
    bb_std = _pos_float(bb_std, 2.0)
    kc_length = _pos_int(kc_length, 20)
    kc_scalar = _pos_float(kc_scalar, 1.5)
    mom_length = _pos_int(mom_length, 12)
    mom_smooth = _pos_int(mom_smooth, 6)
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

    # Calculate Result
    bbd = bbands(close, length=bb_length, std=bb_std, mamode=mamode)
    kch = kc(high, low, close, length=kc_length, scalar=kc_scalar, mamode=mamode, tr=use_tr)
    if bbd is None or kch is None:
        return None

    # Simplify KC and BBAND column names for dynamic access
    bbd.columns = _squeeze_simplify_columns(bbd)
    kch.columns = _squeeze_simplify_columns(kch)

    squeeze = _squeeze_momentum(close, high, low, kch.b, mom_length, mom_smooth, kc_length, mamode, lazybear)

    if squeeze is None:
        return None

    # Classify Squeezes
    squeeze_on = (bbd.l > kch.l) & (bbd.u < kch.u)
    squeeze_off = (bbd.l < kch.l) & (bbd.u > kch.u)
    no_squeeze = ~squeeze_on & ~squeeze_off

    # Convert bool to int before offset to avoid NaN-to-int errors
    if asint:
        squeeze_on = squeeze_on.astype(int)
        squeeze_off = squeeze_off.astype(int)
        no_squeeze = no_squeeze.astype(int)

    # Offset
    squeeze, squeeze_on, squeeze_off, no_squeeze = apply_offset([squeeze, squeeze_on, squeeze_off, no_squeeze], offset)

    squeeze, squeeze_on, squeeze_off, no_squeeze = apply_fill([squeeze, squeeze_on, squeeze_off, no_squeeze], **kwargs)

    # Name and Categorize it
    _props = "" if use_tr else "hlr"
    _props += f"_{bb_length}_{bb_std}_{kc_length}_{kc_scalar}"
    _props += "_LB" if lazybear else ""
    squeeze.name = f"SQZ{_props}"

    data = {
        squeeze.name: squeeze,
        "SQZ_ON": squeeze_on,
        "SQZ_OFF": squeeze_off,
        "SQZ_NO": no_squeeze,
    }
    df = DataFrame(data)
    df.name = squeeze.name
    df.category = squeeze.category = "momentum"

    # Detailed Squeeze Series
    if detailed:
        _squeeze_detailed(df, squeeze, kwargs)

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
