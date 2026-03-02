# Arnaud Legoux Moving Average (ALMA)
from typing import Any, Optional
import numpy as np
from numpy import exp as npExp
from pandas import Series

npNaN = np.nan
from pandas_ta_classic.utils import (
    _finalize,
    _sliding_weighted_ma,
    get_offset,
    verify_series,
)


def alma(
    close: Series,
    length: Optional[int] = None,
    sigma: Optional[float] = None,
    distribution_offset: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Arnaud Legoux Moving Average (ALMA)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 10
    sigma = float(sigma) if sigma and sigma > 0 else 6.0
    distribution_offset = (
        float(distribution_offset)
        if distribution_offset and distribution_offset > 0
        else 0.85
    )
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None:
        return None

    # Pre-Calculations
    m_offset = distribution_offset * (length - 1)
    s = length / sigma
    wtd = np.array(
        [
            npExp(-1 * ((i - m_offset) * (i - m_offset)) / (2 * s * s))
            for i in range(length)
        ]
    )
    w_norm = wtd / wtd.sum()  # normalised weights

    # w_norm[0] = newest, w_norm[L-1] = oldest; sliding_window_view rows are
    # oldest-first, so reverse the weights.
    alma = _sliding_weighted_ma(close, length, w_norm[::-1])

    return _finalize(
        alma,
        offset,
        f"ALMA_{length}_{sigma}_{distribution_offset}",
        "overlap",
        **kwargs,
    )


alma.__doc__ = """Arnaud Legoux Moving Average (ALMA)

The ALMA moving average uses the curve of the Normal (Gauss) distribution, which
can be shifted from 0 to 1. This allows regulating the smoothness and high
sensitivity of the indicator. Sigma is another parameter that is responsible for
the shape of the curve coefficients. This moving average reduces lag of the data
in conjunction with smoothing to reduce noise.

Implemented for Pandas TA by rengel8 based on the source provided below.

Sources:
    https://www.prorealcode.com/prorealtime-indicators/alma-arnaud-legoux-moving-average/

Calculation:
    refer to provided source

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period, window size. Default: 10
    sigma (float): Smoothing value. Default 6.0
    distribution_offset (float): Value to offset the distribution min 0
        (smoother), max 1 (more responsive). Default 0.85
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
