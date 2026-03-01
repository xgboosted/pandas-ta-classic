# -*- coding: utf-8 -*-
# Holt-Winter Channel (HWC)
from math import sqrt as _sqrt
from typing import Any, Optional
from pandas import DataFrame, Series
from pandas_ta_classic.utils import _build_dataframe, get_offset, verify_series


def hwc(
    close: Series,
    na: Optional[float] = None,
    nb: Optional[float] = None,
    nc: Optional[float] = None,
    nd: Optional[float] = None,
    scalar: Optional[float] = None,
    channel_eval: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: Holt-Winter Channel"""
    # Validate Arguments
    na = float(na) if na and na > 0 else 0.2
    nb = float(nb) if nb and nb > 0 else 0.1
    nc = float(nc) if nc and nc > 0 else 0.1
    nd = float(nd) if nd and nd > 0 else 0.1
    scalar = float(scalar) if scalar and scalar > 0 else 1
    channel_eval = bool(channel_eval) if channel_eval and channel_eval else False
    close = verify_series(close)
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result — extract close to numpy to avoid per-bar pandas overhead.
    from numpy import empty as npEmpty

    m = close.size
    c_arr = close.to_numpy()
    last_a = last_v = last_var = 0.0
    last_f = last_price = last_result = c_arr[0]
    result_arr = npEmpty(m)
    upper_arr = npEmpty(m)
    lower_arr = npEmpty(m)
    if channel_eval:
        chan_width_arr = npEmpty(m)
        chan_pct_arr = npEmpty(m)

    for i in range(m):
        F = (1.0 - na) * (last_f + last_v + 0.5 * last_a) + na * c_arr[i]
        V = (1.0 - nb) * (last_v + last_a) + nb * (F - last_f)
        A = (1.0 - nc) * last_a + nc * (V - last_v)
        result_arr[i] = F + V + 0.5 * A

        var = (1.0 - nd) * last_var + nd * (last_price - last_result) * (
            last_price - last_result
        )
        stddev = _sqrt(last_var)
        upper_arr[i] = result_arr[i] + scalar * stddev
        lower_arr[i] = result_arr[i] - scalar * stddev

        if channel_eval:
            width = upper_arr[i] - lower_arr[i]
            chan_width_arr[i] = width
            chan_pct_arr[i] = (c_arr[i] - lower_arr[i]) / width if width != 0 else 0.5

        # update values
        last_price = c_arr[i]
        last_a = A
        last_f = F
        last_v = V
        last_var = var
        last_result = result_arr[i]

    # Aggregate
    hwc = Series(result_arr, index=close.index)
    hwc_upper = Series(upper_arr, index=close.index)
    hwc_lower = Series(lower_arr, index=close.index)
    if channel_eval:
        hwc_width = Series(chan_width_arr, index=close.index)
        hwc_pctwidth = Series(chan_pct_arr, index=close.index)

    series_map = {"HWM": hwc, "HWU": hwc_upper, "HWL": hwc_lower}
    if channel_eval:
        series_map["HWW"] = hwc_width
        series_map["HWPCT"] = hwc_pctwidth

    return _build_dataframe(series_map, "HWC", "volatility", offset, **kwargs)


hwc.__doc__ = """HWC (Holt-Winter Channel)

Channel indicator HWC (Holt-Winters Channel) based on HWMA - a three-parameter
moving average calculated by the method of Holt-Winters.

This version has been implemented for Pandas TA by rengel8 based on a
publication for MetaTrader 5 extended by width and percentage price position
against width of channel.

Sources:
    https://www.mql5.com/en/code/20857

Calculation:
    HWMA[i] = F[i] + V[i] + 0.5 * A[i]
    where..
    F[i] = (1-na) * (F[i-1] + V[i-1] + 0.5 * A[i-1]) + na * Price[i]
    V[i] = (1-nb) * (V[i-1] + A[i-1]) + nb * (F[i] - F[i-1])
    A[i] = (1-nc) * A[i-1] + nc * (V[i] - V[i-1])

    Top = HWMA + Multiplier * StDt
    Bottom = HWMA - Multiplier * StDt
    where..
    StDt[i] = Sqrt(Var[i-1])
    Var[i] = (1-d) * Var[i-1] + nD * (Price[i-1] - HWMA[i-1]) * (Price[i-1] - HWMA[i-1])

Args:
    na - parameter of the equation that describes a smoothed series (from 0 to 1)
    nb - parameter of the equation to assess the trend (from 0 to 1)
    nc - parameter of the equation to assess seasonality (from 0 to 1)
    nd - parameter of the channel equation (from 0 to 1)
    scaler - multiplier for the width of the channel calculated
    channel_eval - boolean to return width and percentage price position against price
    close (pd.Series): Series of 'close's

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method
Returns:
    pd.DataFrame: HWM (Mid), HWU (Upper), HWL (Lower) columns.
"""
