# -*- coding: utf-8 -*-
# Ichimoku Kinko Hyo (ICHIMOKU)
from typing import Any, Optional, Tuple
from pandas import date_range, DataFrame, RangeIndex, Timedelta, Series
from .midprice import midprice
from pandas_ta_classic.utils import apply_offset, get_offset, verify_series


def ichimoku(
    high: Series,
    low: Series,
    close: Series,
    tenkan: Optional[int] = None,
    kijun: Optional[int] = None,
    senkou: Optional[int] = None,
    include_chikou: bool = True,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Tuple[Optional[DataFrame], Optional[DataFrame]]:
    """Indicator: Ichimoku Kinkō Hyō (Ichimoku)"""
    tenkan = int(tenkan) if tenkan and tenkan > 0 else 9
    kijun = int(kijun) if kijun and kijun > 0 else 26
    senkou = int(senkou) if senkou and senkou > 0 else 52
    _length = max(tenkan, kijun, senkou)
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    close = verify_series(close, _length)
    offset = get_offset(offset)
    if not kwargs.get("lookahead", True):
        include_chikou = False

    if high is None or low is None or close is None:
        return None, None

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
    tenkan_sen = apply_offset(tenkan_sen, offset, **kwargs)
    kijun_sen = apply_offset(kijun_sen, offset, **kwargs)
    span_a = apply_offset(span_a, offset, **kwargs)
    span_b = apply_offset(span_b, offset, **kwargs)
    chikou_span = apply_offset(chikou_span, offset, **kwargs)

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

    return ichimokudf, spandf


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

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: Two DataFrames.
        For the visible period: spanA, spanB, tenkan_sen, kijun_sen,
            and chikou_span columns
        For the forward looking period: spanA and spanB columns
"""
