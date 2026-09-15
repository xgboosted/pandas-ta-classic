# Ichimoku Kinko Hyo (ICHIMOKU)
import warnings
from typing import Any

from pandas import DataFrame, RangeIndex, Series, Timedelta, concat, date_range

from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _bool_param, _pos_int, nan_on_short_input

from .midprice import midprice

_ICHIMOKU_TUPLE_DEPRECATION = (
    "ichimoku(as_dataframe=False) returns the legacy (visible, span) tuple, which "
    "is deprecated and will be removed in the next breaking release. Omit "
    "as_dataframe to get the single DataFrame (the default since 0.9.0); add "
    "append_span=True to also get the projected span rows."
)


@nan_on_short_input
def ichimoku(
    high: Series,
    low: Series,
    close: Series,
    tenkan: int | None = None,
    kijun: int | None = None,
    senkou: int | None = None,
    include_chikou: bool = True,
    offset: int | None = None,
    as_dataframe: bool = True,
    append_span: bool = False,
    **kwargs: Any,
) -> tuple[DataFrame | None, DataFrame | None] | DataFrame | None:
    """Indicator: Ichimoku Kinkō Hyō (Ichimoku)"""
    tenkan = _pos_int(tenkan, 9, "tenkan")
    kijun = _pos_int(kijun, 26, "kijun")
    senkou = _pos_int(senkou, 52, "senkou")
    include_chikou = _bool_param(include_chikou, True, "include_chikou")
    as_dataframe = _bool_param(as_dataframe, True, "as_dataframe")
    append_span = _bool_param(append_span, False, "append_span")
    _length = max(tenkan, kijun, senkou)
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    close = verify_series(close, _length)
    offset = get_offset(offset)
    if not _bool_param(kwargs.get("lookahead"), True, "lookahead"):
        include_chikou = False

    return_tuple = not as_dataframe
    if return_tuple:
        warnings.warn(_ICHIMOKU_TUPLE_DEPRECATION, DeprecationWarning, stacklevel=2)

    if high is None or low is None or close is None:
        return (None, None) if return_tuple else None

    # Calculate Result
    tenkan_sen = midprice(high=high, low=low, length=tenkan)
    kijun_sen = midprice(high=high, low=low, length=kijun)
    span_a = 0.5 * (tenkan_sen + kijun_sen)
    span_b = midprice(high=high, low=low, length=senkou)

    # Copy Span A and B values before their shift
    _span_a = span_a[-kijun:].copy()
    _span_b = span_b[-kijun:].copy()

    span_a = span_a.shift(kijun)
    span_b = span_b.shift(kijun)
    chikou_span = close.shift(-kijun)

    # Offset
    tenkan_sen, kijun_sen, span_a, span_b, chikou_span = apply_offset([tenkan_sen, kijun_sen, span_a, span_b, chikou_span], offset)

    tenkan_sen, kijun_sen, span_a, span_b, chikou_span = apply_fill([tenkan_sen, kijun_sen, span_a, span_b, chikou_span], **kwargs)

    # Name and Categorize it
    span_a.name = f"ISA_{tenkan}"
    span_b.name = f"ISB_{kijun}"
    tenkan_sen.name = f"ITS_{tenkan}"
    kijun_sen.name = f"IKS_{kijun}"
    chikou_span.name = f"ICS_{kijun}"

    chikou_span.category = kijun_sen.category = tenkan_sen.category = "trend"
    span_b.category = span_a.category = "trend"

    # Prepare Ichimoku DataFrame
    data = {
        span_a.name: span_a,
        span_b.name: span_b,
        tenkan_sen.name: tenkan_sen,
        kijun_sen.name: kijun_sen,
    }
    if include_chikou:
        data[chikou_span.name] = chikou_span

    ichimokudf = DataFrame(data)
    ichimokudf.name = f"ICHIMOKU_{tenkan}_{kijun}_{senkou}"
    ichimokudf.category = "overlap"

    # Prepare Span DataFrame
    last = close.index[-1]
    if close.index.dtype == "int64":
        ext_index = RangeIndex(start=last + 1, stop=last + kijun + 1)
        spandf = DataFrame(index=ext_index, columns=[span_a.name, span_b.name])
        _span_a.index = _span_b.index = ext_index
    else:
        df_freq = close.index.value_counts().mode()[0]
        tdelta = Timedelta(df_freq, unit="D")
        new_dt = date_range(start=last + tdelta, periods=kijun, freq="B")
        spandf = DataFrame(index=new_dt, columns=[span_a.name, span_b.name])
        _span_a.index = _span_b.index = new_dt

    spandf[span_a.name] = _span_a
    spandf[span_b.name] = _span_b
    spandf.name = f"ICHISPAN_{tenkan}_{kijun}"
    spandf.category = "overlap"

    if return_tuple:
        return ichimokudf, spandf

    # Single-DataFrame return (opt-in via as_dataframe=True). By default the
    # future-dated span rows are omitted (visible period only). append_span=True
    # appends them: concat drops the monkey-patched name/category attrs, so
    # reassign. Span rows carry only ISA/ISB; ITS/IKS/ICS are NaN there.
    if not append_span:
        return ichimokudf

    combined = concat([ichimokudf, spandf], axis=0)
    combined.name = ichimokudf.name
    combined.category = "overlap"
    return combined


ichimoku.__doc__ = """Ichimoku Kinkō Hyō (ichimoku)

Developed Pre WWII as a forecasting model for financial markets.

Sources:
    https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/ichimoku-ich/

Calculation:
    Default Inputs:
        tenkan=9, kijun=26, senkou=52
    MIDPRICE = Midprice
    TENKAN_SEN = MIDPRICE(high, low, close, length=tenkan)
    KIJUN_SEN = MIDPRICE(high, low, close, length=kijun)
    CHIKOU_SPAN = close.shift(-kijun)

    SPAN_A = 0.5 * (TENKAN_SEN + KIJUN_SEN)
    SPAN_A = SPAN_A.shift(kijun)

    SPAN_B = MIDPRICE(high, low, close, length=senkou)
    SPAN_B = SPAN_B.shift(kijun)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    tenkan (int): Tenkan period. Default: 9
    kijun (int): Kijun period. Default: 26
    senkou (int): Senkou period. Default: 52
    include_chikou (bool): Whether to include chikou component. Default: True
    offset (int): How many periods to offset the result. Default: 0
    as_dataframe (bool): Return type selector. True (default) returns a single
        DataFrame. False returns the legacy ``(visible, span)`` tuple and emits
        a DeprecationWarning; the tuple is removed in the next breaking
        release. Default: True
    append_span (bool): Only used when as_dataframe is True. When False
        (default) the returned DataFrame holds the visible period only.
        When True the future-dated span rows (projected Senkou A/B) are
        appended as extra rows. Ignored for the tuple return, which always
        exposes the span separately. Default: False

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    tuple[pd.DataFrame, pd.DataFrame] (as_dataframe is False, deprecated):
        For the visible period: spanA, spanB, tenkan_sen, kijun_sen,
            and chikou_span columns
        For the forward looking period: spanA and spanB columns
    pd.DataFrame (as_dataframe is True, default): the visible period columns. With
        append_span=True the future-dated span rows are appended (only
        spanA/spanB populated there; tenkan_sen/kijun_sen/chikou_span NaN)
"""
