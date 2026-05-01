# -*- coding: utf-8 -*-
# Mesa Sine Wave (MSW)
from typing import Any, Optional

import numpy as np
from pandas import DataFrame, Series

from pandas_ta_classic import Imports
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series


def msw(
    close: Series,
    period: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: Mesa Sine Wave (MSW)

    Identifies cycles using a DFT-based approach from Ehlers (2001).
    Returns two oscillator series: sine and lead (sine + 45°).
    """
    # Validate Arguments
    period = int(period) if period and period > 1 else 5
    close = verify_series(close, period + 1)
    offset = get_offset(offset)

    if close is None:
        return None

    # Tulipy passthrough
    mode_tu = kwargs.get("tulipy", True)
    if Imports["tulipy"] and mode_tu:
        try:
            import tulipy as tu

            result = tu.msw(np.array(close, dtype=float), period=period)
            _size = result[0].size
            _pad = len(close) - _size
            sine_arr = np.concatenate([[np.nan] * _pad, result[0]])
            lead_arr = np.concatenate([[np.nan] * _pad, result[1]])
        except Exception:
            sine_arr, lead_arr = _msw_native(np.array(close, dtype=float), period)
    else:
        sine_arr, lead_arr = _msw_native(np.array(close, dtype=float), period)

    sine = Series(sine_arr, index=close.index)
    lead = Series(lead_arr, index=close.index)

    # Offset
    sine, lead = apply_offset([sine, lead], offset)

    # Handle fills
    sine, lead = apply_fill([sine, lead], **kwargs)

    # Name and Categorize
    _params = f"_{period}"
    sine.name = f"MSW_SINE{_params}"
    lead.name = f"MSW_LEAD{_params}"
    sine.category = lead.category = "cycles"

    df = DataFrame({sine.name: sine, lead.name: lead}, index=close.index)
    df.name = f"MSW{_params}"
    df.category = "cycles"
    return df


def _msw_native(arr: np.ndarray, period: int):
    """Pure numpy Mesa Sine Wave — matches Tulip Indicators algorithm."""
    pi = np.pi
    tpi = 2.0 * pi
    size = len(arr)
    sine = np.full(size, np.nan)
    lead = np.full(size, np.nan)

    j_arr = np.arange(period, dtype=float)
    cos_arr = np.cos(tpi * j_arr / period)
    sin_arr = np.sin(tpi * j_arr / period)

    for i in range(period, size):
        window = arr[i - period + 1 : i + 1][::-1]  # newest first → j=0 is arr[i]
        rp = float(np.dot(window, cos_arr))
        ip = float(np.dot(window, sin_arr))

        if abs(rp) > 0.001:
            phase = np.arctan(ip / rp)
        else:
            phase = (tpi / 2.0) * (-1.0 if ip < 0 else 1.0)

        if rp < 0.0:
            phase += pi
        phase += pi / 2.0
        if phase < 0.0:
            phase += tpi
        if phase > tpi:
            phase -= tpi

        sine[i] = np.sin(phase)
        lead[i] = np.sin(phase + pi / 4.0)

    return sine, lead


msw.__doc__ = """
Mesa Sine Wave (MSW)

Introduced by John F. Ehlers in "Rocket Science For Traders" (2001).
Uses a DFT of the recent ``period`` bars to estimate phase and outputs
two oscillators that help identify cycle turning points.

Sources:
    Tulip Indicators: https://tulipindicators.org/msw
    Ehlers, John F. (2001) Rocket Science For Traders

Args:
    close (pd.Series): Close price series.
    period (int): Lookback period. Default: 5.
    offset (int): Number of periods to offset. Default: 0.

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: Columns MSW_SINE_{period}, MSW_LEAD_{period}.

Example:
    df[['MSW_SINE_5', 'MSW_LEAD_5']] = df.ta.msw()
"""
