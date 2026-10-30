# Bollinger Bands (BBANDS)
from typing import Any

from pandas import DataFrame, Series

from pandas_ta_classic import Imports
from pandas_ta_classic.overlap.ma import ma
from pandas_ta_classic.statistics.stdev import stdev
from pandas_ta_classic.utils import (
    apply_fill,
    apply_offset,
    degenerate_zero,
    get_offset,
    tal_ma,
    verify_series,
)
from pandas_ta_classic.utils._core import (
    _TAL_MA_TYPES,
    _bool_param,
    _name_tokens,
    _nan_where_not_positive,
    _pos_float,
    _pos_int,
    _str_param,
    nan_on_short_input,
)
from pandas_ta_classic.utils._keywords import _MA_OPTIONS


def _bbands_native(close, length, std, ddof, mamode, kwargs):
    """Compute Bollinger Bands without TA-Lib.

    Args:
        close (Series): Close price series.
        length (int): Look-back period.
        std (float): Standard deviation multiplier.
        ddof (int): Delta degrees-of-freedom for stdev.
        mamode (str): MA type for the middle band.
        kwargs (dict): Extra kwargs forwarded to the MA call.

    Returns:
        tuple[Series, Series, Series] | None: ``(lower, mid, upper)`` or
        *None* when any intermediate result is unavailable.
    """
    standard_deviation = stdev(close=close, length=length, ddof=ddof)
    if standard_deviation is None:
        return None
    deviations = std * standard_deviation
    mid = ma(mamode, close, length=length, **kwargs)
    if mid is None:
        return None
    lower = mid - deviations
    upper = mid + deviations
    return lower, mid, upper


@nan_on_short_input
def bbands(
    close: Series,
    length: int | None = None,
    std: float | None = None,
    ddof: int | None = None,
    mamode: str | None = None,
    talib: bool | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> DataFrame | None:
    """Indicator: Bollinger Bands (BBANDS)"""
    # Validate arguments
    length = _pos_int(length, 5, "length", gt=1)
    std = _pos_float(std, 2.0, "std")
    mamode = _str_param(mamode, "sma", "mamode", choices=ma())
    ddof = _pos_int(ddof, 0, "ddof", gt=None, ge=0, lt=length)
    close = verify_series(close, length)
    offset = get_offset(offset)
    mode_talib = _bool_param(talib, False, "talib")

    if close is None:
        return None
    # The fill is for bbands' own output; the inner moving average must not fill
    # its warm-up, or a dema/hma/t3/tema/zlma mid turns that warm-up into band values.
    fill_kwargs = {key: kwargs.pop(key) for key in ("fillna", "fill_method") if key in kwargs}

    # Calculate Result
    # TA-Lib cannot express a non-default ddof, nor an option the chosen moving
    # average reads (sma's min_periods, ema's adjust), nor hma, rma, zlma, ...;
    # run natively instead of ignoring it or failing in tal_ma().
    # The accessor's own options (append, col_names, prefix) are not for the MA.
    ma_options = set(kwargs) & _MA_OPTIONS.get(mamode, frozenset())
    if Imports["talib"] and mode_talib and ddof == 0 and not ma_options and mamode in _TAL_MA_TYPES:
        from talib import BBANDS

        upper, mid, lower = BBANDS(close, length, std, std, tal_ma(mamode))
    else:
        result = _bbands_native(close, length, std, ddof, mamode, kwargs)
        if result is None:
            return None
        lower, mid, upper = result

    # A window whose closes are all equal has a standard deviation of exactly
    # zero, so the band has no width. This needs no malformed bar: a flat close
    # run of `length` bars in otherwise moving, perfectly consistent OHLC does
    # it. With an epsilon width, BBB read 2.2e-16 instead of 0.0, and BBP came
    # out as eps/eps = 1.0 for the exact mamodes -- "price at the top of a band
    # that has no top" -- or exploded to 1e14 for mamode="ema"/"rma", whose mid
    # only converges on the constant and so leaves a non-zero numerator. That
    # is why the mask keys on the width alone and not on both operands, as vhf
    # and marketfi do: here a zero width does *not* imply a zero numerator. The
    # degenerate window reads 0.0, TA-Lib's marker; an exact `ulr` also lets
    # bandwidth reach it on its own.
    ulr = upper - lower
    # A flat close leaves a ~1e-16 standard-deviation residue (and TA-Lib's
    # running STDDEV is not exactly 0 either), so snap it to an exact zero
    # before the width and percent divide on it.
    ulr = ulr.where(~degenerate_zero(ulr), 0.0)
    bandwidth = 100 * ulr / mid
    # A percent of the middle line: NaN where it is at or below zero, with one warning (#345)
    (bandwidth,) = _nan_where_not_positive("bbands", mid, bandwidth)
    percent = ((close - lower) / ulr.where(ulr != 0)).mask(ulr == 0, 0.0)

    # Offset
    lower, mid, upper, bandwidth, percent = apply_offset([lower, mid, upper, bandwidth, percent], offset)

    lower, mid, upper, bandwidth, percent = apply_fill([lower, mid, upper, bandwidth, percent], **fill_kwargs)

    # Name and Categorize it
    _props = f"_{length}_{std}" + _name_tokens(("ddof", ddof, 0))
    lower.name = f"BBL{_props}"
    mid.name = f"BBM{_props}"
    upper.name = f"BBU{_props}"
    bandwidth.name = f"BBB{_props}"
    percent.name = f"BBP{_props}"
    upper.category = lower.category = "volatility"
    mid.category = bandwidth.category = upper.category

    # Prepare DataFrame to return
    data = {
        lower.name: lower,
        mid.name: mid,
        upper.name: upper,
        bandwidth.name: bandwidth,
        percent.name: percent,
    }
    bbandsdf = DataFrame(data)
    bbandsdf.name = f"BBANDS{_props}"
    bbandsdf.category = mid.category

    return bbandsdf


bbands.__doc__ = """Bollinger Bands (BBANDS)

A popular volatility indicator by John Bollinger.

Sources:
    https://www.tradingview.com/wiki/Bollinger_Bands_(BB)

Calculation:
    Default Inputs:
        length=5, std=2, mamode="sma", ddof=0
    EMA = Exponential Moving Average
    SMA = Simple Moving Average
    STDEV = Standard Deviation
    stdev = STDEV(close, length, ddof)
    if "ema":
        MID = EMA(close, length)
    else:
        MID = SMA(close, length)

    LOWER = MID - std * stdev
    UPPER = MID + std * stdev

    BANDWIDTH = 100 * (UPPER - LOWER) / MID
    PERCENT = (close - LOWER) / (UPPER - LOWER)

Args:
    close (pd.Series): Series of 'close's
    length (int): The short period. Default: 5
    std (int): The long period. Default: 2
    ddof (int): Degrees of Freedom to use. Default: 0
    mamode (str): See ```help(ta.ma)```. Default: 'sma'
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: False
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: lower, mid, upper, bandwidth, and percent columns.
"""
