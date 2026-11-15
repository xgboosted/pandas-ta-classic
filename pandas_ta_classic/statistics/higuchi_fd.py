# Higuchi Fractal Dimension (HFD)
from typing import Any

import numpy as np
from pandas import Series

from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _pos_int, nan_on_short_input, skip_leading_nan


@nan_on_short_input
@skip_leading_nan("close")
def higuchi_fd(
    close: Series,
    length: int | None = None,
    kmax: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Higuchi Fractal Dimension (HFD)"""
    # Validate Arguments
    length = _pos_int(length, 30, "length")
    kmax = _pos_int(kmax, 5, "kmax", gt=1)
    if length < 2 * kmax:
        raise ValueError(f"higuchi_fd() needs length >= 2 * kmax, got length={length}, kmax={kmax}")
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None:
        return None

    # Curve length L(k) of every window for k = 1..kmax (Higuchi 1988, eq. 1-2):
    # L_m(k) = (N - 1) / (k^2 * floor((N - m) / k)) * sum_j |x(m + jk) - x(m + (j - 1)k)|,
    # averaged over the k offsets m. A missing bar in the window makes it NaN.
    values = close.to_numpy(dtype=np.float64)
    n = values.size
    result = np.full(n, np.nan, dtype=np.float64)
    windows = np.lib.stride_tricks.sliding_window_view(values, length)
    curve = np.empty((windows.shape[0], kmax), dtype=np.float64)
    for k in range(1, kmax + 1):
        total = np.zeros(windows.shape[0], dtype=np.float64)
        for m in range(k):
            sub = windows[:, m::k]
            steps = sub.shape[1] - 1
            total += np.abs(np.diff(sub, axis=1)).sum(axis=1) * (length - 1) / (k * k * steps)
        curve[:, k - 1] = total / k

    # D is the least-squares slope of ln L(k) against ln(1 / k). A flat
    # window is a straight line, D = 1, as for a line of any other slope. A
    # window with no length at some delays only (a period dividing k) has no
    # power law to fit, so it reads NaN.
    x = -np.log(np.arange(1, kmax + 1, dtype=np.float64))
    x -= x.mean()
    rough = (curve > 0).all(axis=1)
    flat = (curve == 0).all(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.log(curve) @ x / (x @ x)
    result[length - 1 :] = np.where(rough, slope, np.where(flat, 1.0, np.nan))
    hfd = Series(result, index=close.index, dtype=np.float64)

    # Offset
    hfd = apply_offset(hfd, offset)

    hfd = apply_fill(hfd, **kwargs)

    # Name & Category
    hfd.name = f"HFD_{length}_{kmax}"
    hfd.category = "statistics"

    return hfd


higuchi_fd.__doc__ = """Higuchi Fractal Dimension (HFD)

Higuchi's estimate of the fractal dimension D of the price curve over a
rolling window. It measures how rough the path is: a straight trend reads
close to 1, a random walk about 1.5 and white noise about 2. Read it as
D = 2 - H, with H the Hurst exponent: below 1.5 the window trends
(persistent moves), above 1.5 it mean-reverts. It depends only on the shape
of the path, so a shift or a rescaling of the prices leaves it unchanged.

Sources:
    Higuchi, T. (1988). Approach to an irregular time series on the basis of
    the fractal theory. Physica D: Nonlinear Phenomena, 31(2), 277-283.
    https://doi.org/10.1016/0167-2789(88)90081-4

Calculation:
    Default Inputs:
        length=30, kmax=5

    For each window x(1..N) of the last N = length closes, each delay
    k = 1..kmax and each offset m = 1..k:

    L_m(k) = (N - 1) / (k^2 * floor((N - m) / k))
             * SUM(|x(m + j*k) - x(m + (j - 1)*k)|, j = 1..floor((N - m) / k))
    L(k) = MEAN(L_m(k), m = 1..k)
    HFD = slope of the least-squares line of ln L(k) on ln(1 / k)

    A flat window is a straight line and reads 1. A window whose curve
    length is 0 at some delays only (a cycle whose period divides k) reads
    NaN.

Args:
    close (pd.Series): Series of 'close's
    length (int): The window length N. At least 2 * kmax. Default: 30
    kmax (int): The largest delay k. At least 2. Default: 5
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
