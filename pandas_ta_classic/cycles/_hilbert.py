# -*- coding: utf-8 -*-
"""Shared Hilbert Transform helper.

This internal module wraps the numba-accelerated ``_hilbert_transform_loop``
and returns named results as numpy arrays.  Individual HT indicator files
call ``hilbert_result()`` and pick the arrays they need.

The leading underscore keeps ``_meta.py`` from registering this file as an
indicator.
"""
from typing import Dict

import numpy as np
from pandas import Series

from pandas_ta_classic.utils._numba import _hilbert_transform_loop


def hilbert_result(close: Series, ht_start: int = 12) -> Dict[str, np.ndarray]:
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
