# -*- coding: utf-8 -*-
"""Shared Hilbert Transform helper.

This internal module contains the core Hilbert Transform loop and returns
named results as numpy arrays.  Individual HT indicator files call
``hilbert_result()`` and pick the arrays they need.

The leading underscore keeps ``_meta.py`` from registering this file as an
indicator.
"""

from typing import Tuple

import numpy as np
from pandas import Series


def _hilbert_transform_loop(close_arr: np.ndarray, m: int, ht_start: int = 12) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Ehlers Hilbert Transform core loop (TA-Lib compatible).

    Faithfully reproduces the TA-Lib HT algorithm including:
    - Even/odd-equivalent FIR Hilbert Transform
    - DCPeriod rounding via int(smoothPeriod + 0.5)
    - DCPhase edge-case handling (incremental adjustment)
    - 4-step trend mode detection (crossover, daysInTrend, phase range,
      price-trendline divergence)

    Returns ``(smooth_period, dc_phase, in_phase, quadrature,
               sine, lead_sine, trend_mode, trendline)`` arrays.
    """
    smooth_price = np.zeros(m)
    detrend = np.zeros(m)
    i1 = np.zeros(m)  # InPhase
    q1 = np.zeros(m)  # Quadrature
    ji = np.zeros(m)
    jq = np.zeros(m)
    i2 = np.zeros(m)
    q2 = np.zeros(m)
    re_ = np.zeros(m)
    im_ = np.zeros(m)
    period = np.zeros(m)
    smooth_period_arr = np.full(m, np.nan)
    dc_phase_arr = np.full(m, np.nan)
    in_phase_arr = np.full(m, np.nan)
    quad_arr = np.full(m, np.nan)
    sine_arr = np.full(m, np.nan)
    lead_sine_arr = np.full(m, np.nan)
    trend_mode_arr = np.full(m, np.nan)
    trendline_arr = np.full(m, np.nan)

    # iTrend delay line (WMA(4) of SMA(dcPeriod))
    it_trend = np.zeros(m)

    # Pre-compute sin/cos lookup tables for DC Phase inner loop.
    # dc_period_int is clamped to [1, 50], so we need tables up to 50.
    _MAX_P = 51
    sin_table = np.zeros((_MAX_P, _MAX_P))
    cos_table = np.zeros((_MAX_P, _MAX_P))
    for p in range(1, _MAX_P):
        for j in range(p):
            angle = 2.0 * np.pi * j / p
            sin_table[p, j] = np.sin(angle)
            cos_table[p, j] = np.cos(angle)

    # Pre-compute cumulative sum for iTrend SMA (O(1) per bar).
    close_cumsum = np.zeros(m + 1)
    for ci in range(m):
        close_cumsum[ci + 1] = close_cumsum[ci] + close_arr[ci]

    # State for trend mode detection (matches TA-Lib)
    days_in_trend = 0
    prev_sine = 0.0
    prev_lead_sine = 0.0
    prev_dc_phase = 0.0

    for i in range(m):
        # WMA(4) smoothing — only stored from bar ht_start onwards
        if i >= ht_start and i >= 3:
            smooth_price[i] = (
                4.0 * close_arr[i]
                + 3.0 * close_arr[i - 1]
                + 2.0 * close_arr[i - 2]
                + close_arr[i - 3]
            ) / 10.0

        if i < ht_start:
            period[i] = 0.0
            smooth_period_arr[i] = 0.0
            continue

        adj = 0.075 * period[i - 1] + 0.54

        # Detrend (4-tap FIR)
        detrend[i] = (
            0.0962 * smooth_price[i]
            + 0.5769 * smooth_price[i - 2]
            - 0.5769 * smooth_price[i - 4]
            - 0.0962 * smooth_price[i - 6]
        ) * adj

        # InPhase and Quadrature
        q1[i] = (
            0.0962 * detrend[i]
            + 0.5769 * detrend[i - 2]
            - 0.5769 * detrend[i - 4]
            - 0.0962 * detrend[i - 6]
        ) * adj
        i1[i] = detrend[i - 3]

        # Advance the phase of I1 and Q1 by 90 degrees
        ji[i] = (
            0.0962 * i1[i]
            + 0.5769 * i1[i - 2]
            - 0.5769 * i1[i - 4]
            - 0.0962 * i1[i - 6]
        ) * adj
        jq[i] = (
            0.0962 * q1[i]
            + 0.5769 * q1[i - 2]
            - 0.5769 * q1[i - 4]
            - 0.0962 * q1[i - 6]
        ) * adj

        # Phasor addition for 3-bar averaging
        i2[i] = i1[i] - jq[i]
        q2[i] = q1[i] + ji[i]

        # Smooth the I and Q components
        i2[i] = 0.2 * i2[i] + 0.8 * i2[i - 1]
        q2[i] = 0.2 * q2[i] + 0.8 * q2[i - 1]

        # Homodyne Discriminator
        re_[i] = i2[i] * i2[i - 1] + q2[i] * q2[i - 1]
        im_[i] = i2[i] * q2[i - 1] - q2[i] * i2[i - 1]
        re_[i] = 0.2 * re_[i] + 0.8 * re_[i - 1]
        im_[i] = 0.2 * im_[i] + 0.8 * im_[i - 1]

        if im_[i] != 0.0 and re_[i] != 0.0:
            period[i] = 360.0 / (np.arctan(im_[i] / re_[i]) * 180.0 / np.pi)
        else:
            period[i] = period[i - 1]

        if period[i] > 1.5 * period[i - 1]:
            period[i] = 1.5 * period[i - 1]
        if period[i] < 0.67 * period[i - 1]:
            period[i] = 0.67 * period[i - 1]
        if period[i] < 6.0:
            period[i] = 6.0
        if period[i] > 50.0:
            period[i] = 50.0

        period[i] = 0.2 * period[i] + 0.8 * period[i - 1]
        smooth_period_arr[i] = 0.33 * period[i] + 0.67 * smooth_period_arr[i - 1]

        # DC Phase — TA-Lib uses int(smoothPeriod + 0.5) for rounding
        sp = smooth_period_arr[i]
        dc_period_int = max(int(sp + 0.5), 1)
        real_part = 0.0
        imag_part = 0.0
        for j in range(dc_period_int):
            if i - j >= 0:
                sp_val = smooth_price[i - j]
                real_part += sin_table[dc_period_int, j] * sp_val
                imag_part += cos_table[dc_period_int, j] * sp_val

        # TA-Lib DCPhase: fabs(imagPart) > 0 -> atan; else adjust previous
        abs_imag = abs(imag_part)
        if abs_imag > 0.0:
            dc_phase_val = np.arctan(real_part / imag_part) * 180.0 / np.pi
        else:
            dc_phase_val = prev_dc_phase
            if real_part < 0.0:
                dc_phase_val -= 90.0
            elif real_part > 0.0:
                dc_phase_val += 90.0

        dc_phase_val += 90.0
        if sp > 0.0:
            dc_phase_val += 360.0 / sp
        if imag_part < 0.0:
            dc_phase_val += 180.0
        if dc_phase_val > 315.0:
            dc_phase_val -= 360.0

        dc_phase_arr[i] = dc_phase_val
        in_phase_arr[i] = i1[i]
        quad_arr[i] = q1[i]

        # Sine / LeadSine
        sine_val = np.sin(dc_phase_val * np.pi / 180.0)
        lead_sine_val = np.sin((dc_phase_val + 45.0) * np.pi / 180.0)
        sine_arr[i] = sine_val
        lead_sine_arr[i] = lead_sine_val

        # Instantaneous Trendline (ITrend)
        dc_per = max(int(sp + 0.5), 1)
        start_idx = i - dc_per + 1
        if start_idx < 0:
            start_idx = 0
        it_trend[i] = (close_cumsum[i + 1] - close_cumsum[start_idx]) / dc_per
        trendline_arr[i] = (
            4.0 * it_trend[i]
            + 3.0 * it_trend[i - 1]
            + 2.0 * (it_trend[i - 2] if i >= 2 else it_trend[0])
            + (it_trend[i - 3] if i >= 3 else it_trend[0])
        ) / 10.0

        # ----- Trend Mode (TA-Lib 4-step algorithm) -----
        trend = 1

        # Step 1: Sine/LeadSine crossover resets trend counter
        if (sine_val > lead_sine_val and prev_sine <= prev_lead_sine) or (
            sine_val < lead_sine_val and prev_sine >= prev_lead_sine
        ):
            days_in_trend = 0
            trend = 0

        days_in_trend += 1

        # Step 2: Not enough bars since crossover -> cycle mode
        if days_in_trend < 0.5 * sp:
            trend = 0

        # Step 3: Phase change in expected cycle range -> cycle mode
        phase_diff = dc_phase_val - prev_dc_phase
        if (
            sp != 0.0
            and phase_diff > 0.67 * 360.0 / sp
            and phase_diff < 1.5 * 360.0 / sp
        ):
            trend = 0

        # Step 4: Price far from trendline -> trend mode override
        if (
            trendline_arr[i] != 0.0
            and abs((smooth_price[i] - trendline_arr[i]) / trendline_arr[i]) >= 0.015
        ):
            trend = 1

        trend_mode_arr[i] = float(trend)

        # Update previous-bar state
        prev_sine = sine_val
        prev_lead_sine = lead_sine_val
        prev_dc_phase = dc_phase_val

    return (
        smooth_period_arr,
        dc_phase_arr,
        in_phase_arr,
        quad_arr,
        sine_arr,
        lead_sine_arr,
        trend_mode_arr,
        trendline_arr,
    )


def hilbert_result(close: Series, ht_start: int = 12) -> dict:
    """Run the Hilbert Transform and return all intermediate arrays.

    Args:
        close: Series of close prices.
        ht_start: Bar index where the Hilbert computation begins.
            TA-Lib uses 12 for HT_DCPERIOD/HT_PHASOR (lookback 32)
            and 37 for HT_DCPHASE/HT_SINE/HT_TRENDMODE/HT_TRENDLINE
            (lookback 63).

    Returns:
        Dict with keys: ``smooth_period``, ``dc_phase``, ``in_phase``,
        ``quadrature``, ``sine``, ``lead_sine``, ``trend_mode``,
        ``trendline``.
    """
    c_arr = close.to_numpy(dtype=float)
    m = c_arr.shape[0]

    (
        smooth_period,
        dc_phase,
        in_phase,
        quadrature,
        sine,
        lead_sine,
        trend_mode,
        trendline,
    ) = _hilbert_transform_loop(c_arr, m, ht_start)

    return {
        "smooth_period": smooth_period,
        "dc_phase": dc_phase,
        "in_phase": in_phase,
        "quadrature": quadrature,
        "sine": sine,
        "lead_sine": lead_sine,
        "trend_mode": trend_mode,
        "trendline": trendline,
    }
