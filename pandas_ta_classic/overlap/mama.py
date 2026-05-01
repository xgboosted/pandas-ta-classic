# -*- coding: utf-8 -*-
# MESA Adaptive Moving Average (MAMA)
from typing import Any, Optional, Tuple
import numpy as np
from pandas import DataFrame, Series
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series


def _mama_loop(
    close_arr: np.ndarray,
    m: int,
    fastlimit: float,
    slowlimit: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """MAMA / FAMA with inline Hilbert Transform (TA-Lib exact)."""
    mama_out = np.full(m, np.nan)
    fama_out = np.full(m, np.nan)

    a = 0.0962
    b = 0.5769
    rad2deg = 180.0 / np.pi

    # Hilbert ring-buffer state (4 components)
    det_odd = np.zeros(3)
    det_even = np.zeros(3)
    prev_det_odd = prev_det_even = 0.0
    prev_det_in_odd = prev_det_in_even = 0.0
    detrender = 0.0

    q1_odd = np.zeros(3)
    q1_even = np.zeros(3)
    prev_q1_odd = prev_q1_even = 0.0
    prev_q1_in_odd = prev_q1_in_even = 0.0
    Q1 = 0.0

    ji_odd = np.zeros(3)
    ji_even = np.zeros(3)
    prev_ji_odd = prev_ji_even = 0.0
    prev_ji_in_odd = prev_ji_in_even = 0.0
    jI = 0.0

    jq_odd = np.zeros(3)
    jq_even = np.zeros(3)
    prev_jq_odd = prev_jq_even = 0.0
    prev_jq_in_odd = prev_jq_in_even = 0.0
    jQ = 0.0

    period = 0.0
    prev_i2 = prev_q2 = 0.0
    re = im = 0.0
    mama_val = fama_val = 0.0
    prev_phase = 0.0
    hilbert_idx = 0

    i1_for_odd_prev2 = i1_for_odd_prev3 = 0.0
    i1_for_even_prev2 = i1_for_even_prev3 = 0.0

    lookback = 32
    if m < 4:
        return mama_out, fama_out

    # WMA init: accumulate first 3 bars
    trailing_wma_idx = 0
    period_wma_sub = close_arr[0] + close_arr[1] + close_arr[2]
    period_wma_sum = close_arr[0] * 1.0 + close_arr[1] * 2.0 + close_arr[2] * 3.0
    trailing_wma_value = 0.0

    # WMA warmup loop: 9 more bars (indices 3..11)
    smoothed_value = 0.0
    for idx in range(3, 12):
        if idx >= m:
            return mama_out, fama_out
        new_price = close_arr[idx]
        period_wma_sub += new_price
        period_wma_sub -= trailing_wma_value
        period_wma_sum += new_price * 4.0
        trailing_wma_value = close_arr[trailing_wma_idx]
        trailing_wma_idx += 1
        smoothed_value = period_wma_sum * 0.1
        period_wma_sum -= period_wma_sub

    # Main loop (from bar 12 onward)
    today = 12
    while today < m:
        adjusted_prev_period = 0.075 * period + 0.54
        today_value = close_arr[today]

        # DO_PRICE_WMA
        period_wma_sub += today_value
        period_wma_sub -= trailing_wma_value
        period_wma_sum += today_value * 4.0
        trailing_wma_value = close_arr[trailing_wma_idx]
        trailing_wma_idx += 1
        smoothed_value = period_wma_sum * 0.1
        period_wma_sum -= period_wma_sub

        if (today % 2) == 0:
            # EVEN bar
            ht = a * smoothed_value
            detrender = -det_even[hilbert_idx]
            det_even[hilbert_idx] = ht
            detrender += ht - prev_det_even
            prev_det_even = b * prev_det_in_even
            detrender += prev_det_even
            prev_det_in_even = smoothed_value
            detrender *= adjusted_prev_period

            ht = a * detrender
            Q1 = -q1_even[hilbert_idx]
            q1_even[hilbert_idx] = ht
            Q1 += ht - prev_q1_even
            prev_q1_even = b * prev_q1_in_even
            Q1 += prev_q1_even
            prev_q1_in_even = detrender
            Q1 *= adjusted_prev_period

            ht = a * i1_for_even_prev3
            jI = -ji_even[hilbert_idx]
            ji_even[hilbert_idx] = ht
            jI += ht - prev_ji_even
            prev_ji_even = b * prev_ji_in_even
            jI += prev_ji_even
            prev_ji_in_even = i1_for_even_prev3
            jI *= adjusted_prev_period

            ht = a * Q1
            jQ = -jq_even[hilbert_idx]
            jq_even[hilbert_idx] = ht
            jQ += ht - prev_jq_even
            prev_jq_even = b * prev_jq_in_even
            jQ += prev_jq_even
            prev_jq_in_even = Q1
            jQ *= adjusted_prev_period

            hilbert_idx += 1
            if hilbert_idx == 3:
                hilbert_idx = 0

            Q2 = 0.2 * (Q1 + jI) + 0.8 * prev_q2
            I2 = 0.2 * (i1_for_even_prev3 - jQ) + 0.8 * prev_i2

            i1_for_odd_prev3 = i1_for_odd_prev2
            i1_for_odd_prev2 = detrender

            if i1_for_even_prev3 != 0.0:
                phase = np.arctan(Q1 / i1_for_even_prev3) * rad2deg
            else:
                phase = 0.0
        else:
            # ODD bar
            ht = a * smoothed_value
            detrender = -det_odd[hilbert_idx]
            det_odd[hilbert_idx] = ht
            detrender += ht - prev_det_odd
            prev_det_odd = b * prev_det_in_odd
            detrender += prev_det_odd
            prev_det_in_odd = smoothed_value
            detrender *= adjusted_prev_period

            ht = a * detrender
            Q1 = -q1_odd[hilbert_idx]
            q1_odd[hilbert_idx] = ht
            Q1 += ht - prev_q1_odd
            prev_q1_odd = b * prev_q1_in_odd
            Q1 += prev_q1_odd
            prev_q1_in_odd = detrender
            Q1 *= adjusted_prev_period

            ht = a * i1_for_odd_prev3
            jI = -ji_odd[hilbert_idx]
            ji_odd[hilbert_idx] = ht
            jI += ht - prev_ji_odd
            prev_ji_odd = b * prev_ji_in_odd
            jI += prev_ji_odd
            prev_ji_in_odd = i1_for_odd_prev3
            jI *= adjusted_prev_period

            ht = a * Q1
            jQ = -jq_odd[hilbert_idx]
            jq_odd[hilbert_idx] = ht
            jQ += ht - prev_jq_odd
            prev_jq_odd = b * prev_jq_in_odd
            jQ += prev_jq_odd
            prev_jq_in_odd = Q1
            jQ *= adjusted_prev_period

            Q2 = 0.2 * (Q1 + jI) + 0.8 * prev_q2
            I2 = 0.2 * (i1_for_odd_prev3 - jQ) + 0.8 * prev_i2

            i1_for_even_prev3 = i1_for_even_prev2
            i1_for_even_prev2 = detrender

            if i1_for_odd_prev3 != 0.0:
                phase = np.arctan(Q1 / i1_for_odd_prev3) * rad2deg
            else:
                phase = 0.0

        # Delta Phase -> Alpha
        delta_phase = prev_phase - phase
        prev_phase = phase
        if delta_phase < 1.0:
            delta_phase = 1.0

        if delta_phase > 1.0:
            alpha = fastlimit / delta_phase
            if alpha < slowlimit:
                alpha = slowlimit
        else:
            alpha = fastlimit

        # MAMA / FAMA
        mama_val = alpha * today_value + (1.0 - alpha) * mama_val
        fama_val = 0.5 * alpha * mama_val + (1.0 - 0.5 * alpha) * fama_val

        if today >= lookback:
            mama_out[today] = mama_val
            fama_out[today] = fama_val

        # Period update (homodyne discriminator)
        re = 0.2 * (I2 * prev_i2 + Q2 * prev_q2) + 0.8 * re
        im = 0.2 * (I2 * prev_q2 - Q2 * prev_i2) + 0.8 * im
        prev_q2 = Q2
        prev_i2 = I2

        prev_period = period
        if im != 0.0 and re != 0.0:
            period = 360.0 / (np.arctan(im / re) * rad2deg)

        if period > 1.5 * prev_period:
            period = 1.5 * prev_period
        if period < 0.67 * prev_period:
            period = 0.67 * prev_period
        if period < 6.0:
            period = 6.0
        elif period > 50.0:
            period = 50.0

        period = 0.2 * period + 0.8 * prev_period

        today += 1

    return mama_out, fama_out


def mama(
    close: Series,
    fastlimit: Optional[float] = None,
    slowlimit: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: MESA Adaptive Moving Average (MAMA)"""
    # Validate Arguments
    fastlimit = float(fastlimit) if fastlimit and fastlimit > 0 else 0.5
    slowlimit = float(slowlimit) if slowlimit and slowlimit > 0 else 0.05
    close = verify_series(close)
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result
    c_arr = close.to_numpy(dtype=float)
    m = c_arr.shape[0]
    mama_arr, fama_arr = _mama_loop(c_arr, m, fastlimit, slowlimit)
    mama_s = Series(mama_arr, index=close.index)
    fama_s = Series(fama_arr, index=close.index)

    # Offset
    mama_s, fama_s = apply_offset([mama_s, fama_s], offset)

    # Handle fills
    mama_s, fama_s = apply_fill([mama_s, fama_s], **kwargs)

    # Name and Categorize it
    _params = f"_{fastlimit}_{slowlimit}"
    mama_s.name = f"MAMA{_params}"
    fama_s.name = f"FAMA{_params}"

    data = {mama_s.name: mama_s, fama_s.name: fama_s}
    df = DataFrame(data)
    df.name = f"MAMA{_params}"
    df.category = "overlap"

    return df


mama.__doc__ = """MESA Adaptive Moving Average (MAMA)

MAMA adapts to price movement based on the rate of change of the Hilbert
Transform phase.  It uses a fast limit and slow limit to bound the
smoothing factor.  FAMA (Following Adaptive Moving Average) is a further
smoothed version of MAMA.

Sources:
    John F. Ehlers, "MESA and Trading Market Cycles"

Calculation:
    Default Inputs:
        fastlimit=0.5, slowlimit=0.05
    alpha = fastlimit / delta_phase  (clamped to [slowlimit, fastlimit])
    MAMA = alpha * close + (1 - alpha) * MAMA[1]
    FAMA = 0.5 * alpha * MAMA + (1 - 0.5 * alpha) * FAMA[1]

Args:
    close (pd.Series): Series of 'close's
    fastlimit (float): Upper bound for the adaptive alpha. Default: 0.5
    slowlimit (float): Lower bound for the adaptive alpha. Default: 0.05
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: MAMA and FAMA columns.
"""
