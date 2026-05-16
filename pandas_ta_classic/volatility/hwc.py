# -*- coding: utf-8 -*-
# Holt-Winter Channel (HWC)
from typing import Any, Optional
import numpy as np
from pandas import DataFrame, Series
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._njit import njit


@njit(cache=True)
def _hwc_loop(c_arr, m, na, nb, nc, nd, scalar):
    result_arr = np.empty(m)
    upper_arr = np.empty(m)
    lower_arr = np.empty(m)
    last_a = 0.0
    last_v = 0.0
    last_var = 0.0
    last_f = c_arr[0]
    last_price = c_arr[0]
    last_result = c_arr[0]
    for i in range(m):
        F = (1.0 - na) * (last_f + last_v + 0.5 * last_a) + na * c_arr[i]
        V = (1.0 - nb) * (last_v + last_a) + nb * (F - last_f)
        A = (1.0 - nc) * last_a + nc * (V - last_v)
        result_arr[i] = F + V + 0.5 * A
        var = (1.0 - nd) * last_var + nd * (last_price - last_result) * (
            last_price - last_result
        )
        stddev = last_var**0.5
        upper_arr[i] = result_arr[i] + scalar * stddev
        lower_arr[i] = result_arr[i] - scalar * stddev
        last_price = c_arr[i]
        last_a = A
        last_f = F
        last_v = V
        last_var = var
        last_result = result_arr[i]
    return result_arr, upper_arr, lower_arr


def _hwc_build_df(hwc_s, upper_s, lower_s, width_s, pctwidth_s, channel_eval):
    """Assemble and name the HWC result DataFrame.

    Args:
        hwc_s (Series): Centre channel series.
        upper_s (Series): Upper band series.
        lower_s (Series): Lower band series.
        width_s (Series | None): Channel width (only when *channel_eval* is True).
        pctwidth_s (Series | None): Percent width (only when *channel_eval* is True).
        channel_eval (bool): Whether to include width / pctwidth columns.

    Returns:
        DataFrame: Named HWC result frame.
    """
    hwc_s.name = "HWM"
    upper_s.name = "HWU"
    lower_s.name = "HWL"
    hwc_s.category = upper_s.category = lower_s.category = "volatility"

    data = {hwc_s.name: hwc_s, upper_s.name: upper_s, lower_s.name: lower_s}
    if channel_eval:
        width_s.name = "HWW"
        pctwidth_s.name = "HWPCT"
        data[width_s.name] = width_s
        data[pctwidth_s.name] = pctwidth_s

    df = DataFrame(data)
    df.name = "HWC"
    df.category = hwc_s.category
    return df


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
    if close is None:
        return None
    offset = get_offset(offset)

    # Calculate Result
    m = close.size
    c_arr = close.to_numpy(dtype=float)
    result_arr, upper_arr, lower_arr = _hwc_loop(c_arr, m, na, nb, nc, nd, scalar)

    # Aggregate
    hwc_s = Series(result_arr, index=close.index)
    hwc_upper = Series(upper_arr, index=close.index)
    hwc_lower = Series(lower_arr, index=close.index)
    hwc_width = hwc_pctwidth = None
    if channel_eval:
        hwc_width = Series(upper_arr - lower_arr, index=close.index)
        hwc_pctwidth = Series(
            (c_arr - lower_arr) / (upper_arr - lower_arr), index=close.index
        )

    # Offset
    hwc_s, hwc_upper, hwc_lower = apply_offset([hwc_s, hwc_upper, hwc_lower], offset)
    if channel_eval:
        hwc_width, hwc_pctwidth = apply_offset([hwc_width, hwc_pctwidth], offset)

    # Handle fills
    hwc_s, hwc_upper, hwc_lower = apply_fill([hwc_s, hwc_upper, hwc_lower], **kwargs)
    if channel_eval:
        hwc_width, hwc_pctwidth = apply_fill([hwc_width, hwc_pctwidth], **kwargs)

    return _hwc_build_df(
        hwc_s, hwc_upper, hwc_lower, hwc_width, hwc_pctwidth, channel_eval
    )


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
