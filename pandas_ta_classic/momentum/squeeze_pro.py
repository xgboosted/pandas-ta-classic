# Squeeze Pro (SQUEEZE_PRO)
import warnings
from typing import Any

from pandas import DataFrame, Series

from pandas_ta_classic.momentum.mom import mom
from pandas_ta_classic.momentum.squeeze import _squeeze_detailed, _squeeze_simplify_columns, _squeeze_use_tr
from pandas_ta_classic.overlap.ma import ma
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _bool_param, _name_tokens, _pos_float, _pos_int, _str_param, nan_on_short_input
from pandas_ta_classic.volatility.bbands import bbands
from pandas_ta_classic.volatility.kc import kc


@nan_on_short_input
def squeeze_pro(
    high: Series,
    low: Series,
    close: Series,
    bb_length: int | None = None,
    bb_std: float | None = None,
    kc_length: int | None = None,
    kc_scalar_wide: float | None = None,
    kc_scalar_normal: float | None = None,
    kc_scalar_narrow: float | None = None,
    mom_length: int | None = None,
    mom_smooth: int | None = None,
    use_tr: bool | None = None,
    mamode: str | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> DataFrame | None:
    """Indicator: Squeeze Momentum (SQZ) PRO"""
    # Validate arguments
    bb_length = _pos_int(bb_length, 20, "bb_length")
    bb_std = _pos_float(bb_std, 2.0, "bb_std")
    kc_length = _pos_int(kc_length, 20, "kc_length")
    kc_scalar_wide = _pos_float(kc_scalar_wide, 2, "kc_scalar_wide")
    kc_scalar_normal = _pos_float(kc_scalar_normal, 1.5, "kc_scalar_normal")
    kc_scalar_narrow = _pos_float(kc_scalar_narrow, 1, "kc_scalar_narrow")
    mom_length = _pos_int(mom_length, 12, "mom_length")
    mom_smooth = _pos_int(mom_smooth, 6, "mom_smooth")

    _length = max(bb_length, kc_length, mom_length, mom_smooth)
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    close = verify_series(close, _length)
    offset = get_offset(offset)

    if not (kc_scalar_wide > kc_scalar_normal > kc_scalar_narrow):
        raise ValueError(
            "squeeze_pro() needs kc_scalar_wide > kc_scalar_normal > kc_scalar_narrow, "
            f"got {kc_scalar_wide}, {kc_scalar_normal}, {kc_scalar_narrow}"
        )
    if high is None or low is None or close is None:
        return None

    use_tr = _squeeze_use_tr(use_tr, kwargs, "squeeze_pro")
    asint = _bool_param(kwargs.pop("asint", None), True, "asint")
    detailed = _bool_param(kwargs.pop("detailed", None), False, "detailed")
    mamode = _str_param(mamode, "sma", "mamode", choices=ma())

    # Calculate Result
    # Only the bands are read here; bbands' warning about a non-positive middle
    # line concerns its bandwidth column, which this indicator does not return (#345).
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"bbands\(\) reads NaN", category=UserWarning)
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
    squeeze = ma(mamode, momo, length=mom_smooth)

    # Classify Squeezes
    squeeze_on_wide = (bbd.l > kch_wide.l) & (bbd.u < kch_wide.u)
    squeeze_on_normal = (bbd.l > kch_normal.l) & (bbd.u < kch_normal.u)
    squeeze_on_narrow = (bbd.l > kch_narrow.l) & (bbd.u < kch_narrow.u)
    squeeze_off_wide = (bbd.l < kch_wide.l) & (bbd.u > kch_wide.u)
    no_squeeze = ~squeeze_on_wide & ~squeeze_off_wide

    # On warm-up bars the bands are NaN, every comparison is False, and
    # no_squeeze would wrongly read 1 (as in squeeze). Mask the flags to NaN there.
    defined = bbd.l.notna() & bbd.u.notna()
    for kc_ in (kch_wide, kch_normal, kch_narrow):
        defined &= kc_.l.notna() & kc_.u.notna()
    squeeze_on_wide, squeeze_on_normal, squeeze_on_narrow, squeeze_off_wide, no_squeeze = (
        flag.where(defined) for flag in (squeeze_on_wide, squeeze_on_normal, squeeze_on_narrow, squeeze_off_wide, no_squeeze)
    )

    # Convert bool to float (not int): NaN must survive on the warm-up bars.
    if asint:
        squeeze_on_wide = squeeze_on_wide.astype(float)
        squeeze_on_normal = squeeze_on_normal.astype(float)
        squeeze_on_narrow = squeeze_on_narrow.astype(float)
        squeeze_off_wide = squeeze_off_wide.astype(float)
        no_squeeze = no_squeeze.astype(float)

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

    # The detailed columns derive from the momentum before the fill (apply_fill fills in place).
    momentum = squeeze.copy() if detailed else None
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
    _props += _name_tokens(("mom_length", mom_length, 12), ("mom_smooth", mom_smooth, 6))
    squeeze.name = f"SQZPRO{_props}"

    # The flags compare the bands with the channels, so their settings name them too.
    _flags = _name_tokens(
        ("bb_length", bb_length, 20),
        ("bb_std", bb_std, 2.0),
        ("kc_length", kc_length, 20),
        ("kc_scalar_wide", kc_scalar_wide, 2),
        ("kc_scalar_normal", kc_scalar_normal, 1.5),
        ("kc_scalar_narrow", kc_scalar_narrow, 1),
        ("use_tr", use_tr, True),
    )
    data = {
        squeeze.name: squeeze,
        f"SQZPRO_ON_WIDE{_flags}": squeeze_on_wide,
        f"SQZPRO_ON_NORMAL{_flags}": squeeze_on_normal,
        f"SQZPRO_ON_NARROW{_flags}": squeeze_on_narrow,
        f"SQZPRO_OFF{_flags}": squeeze_off_wide,
        f"SQZPRO_NO{_flags}": no_squeeze,
    }
    df = DataFrame(data)
    df.name = squeeze.name
    df.category = squeeze.category = "momentum"

    # Detailed Squeeze Series
    if detailed:
        _squeeze_detailed(df, momentum, kwargs, prefix="SQZPRO_")

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
    MA = the moving average mamode selects, see help(ta.ma)
    TR = True Range

    RANGE = TR(high, low, close) if using_tr else high - low
    BB_LOW, BB_MID, BB_HIGH = BB(close, bb_length, std=bb_std)
    KC_LOW_WIDE, KC_MID_WIDE, KC_HIGH_WIDE = KC(high, low, close, kc_length, kc_scalar_wide, TR)
    KC_LOW_NORMAL, KC_MID_NORMAL, KC_HIGH_NORMAL = KC(high, low, close, kc_length, kc_scalar_normal, TR)
    KC_LOW_NARROW, KC_MID_NARROW, KC_HIGH_NARROW = KC(high, low, close, kc_length, kc_scalar_narrow, TR)

    MOMO = MOM(close, mom_length)
    SQZPRO = MA(mamode, MOMO, mom_smooth)

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
    use_tr (bool): Use True Range for Keltner Channels. Default: True
    mamode (str): The MA of the bands and of the momentum smoothing.
        See ``help(ta.ma)``. Default: "sma"
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    tr (value, optional): Older name of ``use_tr``; passing both raises.
    asint (value, optional): Flags as numbers (1.0/0.0) instead of bool; NaN on
        warm-up bars, where the bands do not exist yet. Default: True
    detailed (value, optional): Return additional variations of SQZ for
        visualization. Default: False
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: SQZPRO, SQZPRO_ON_WIDE, SQZPRO_ON_NORMAL, SQZPRO_ON_NARROW, SQZPRO_OFF, SQZPRO_NO columns by default. More
        detailed columns if 'detailed' kwarg is True.
"""
