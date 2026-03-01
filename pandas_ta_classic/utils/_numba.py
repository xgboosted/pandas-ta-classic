# -*- coding: utf-8 -*-
"""Numba-accelerated loop kernels for iterative indicators.

When numba is installed (``pip install pandas-ta-classic[performance]``),
each function is compiled to machine code via ``@njit(cache=True)``.
When numba is *not* installed the same functions run as plain CPython —
numerically identical, just slower.
"""
from typing import Tuple

import numpy as np

try:
    from numba import njit
except ImportError:

    def njit(*args, **kwargs):  # type: ignore[misc]
        """No-op decorator that mimics ``numba.njit`` when numba is absent."""

        def _wrap(f):
            return f

        return _wrap if not args or not callable(args[0]) else args[0]


# ---------------------------------------------------------------------------
# 1. RSX  (momentum/rsx.py)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _rsx_loop(c_arr: np.ndarray, length: int, m: int) -> np.ndarray:
    """Core RSX iterative filter.  Returns result array of length *m*."""
    result = np.full(m, np.nan)
    result[length - 1] = 0.0

    # State variables
    vC = 0.0
    v1C = 0.0
    v4 = 0.0
    v8 = 0.0
    v10 = 0.0
    v14 = 0.0
    v18 = 0.0
    v20 = 0.0

    f0 = 0.0
    f8 = 0.0
    f10 = 0.0
    f18 = 0.0
    f20 = 0.0
    f28 = 0.0
    f30 = 0.0
    f38 = 0.0
    f40 = 0.0
    f48 = 0.0
    f50 = 0.0
    f58 = 0.0
    f60 = 0.0
    f68 = 0.0
    f70 = 0.0
    f78 = 0.0
    f80 = 0.0
    f88 = 0.0
    f90 = 0.0

    for i in range(length, m):
        if f90 == 0:
            f90 = 1.0
            f0 = 0.0
            if length - 1.0 >= 5:
                f88 = length - 1.0
            else:
                f88 = 5.0
            f8 = 100.0 * c_arr[i]
            f18 = 3.0 / (length + 2.0)
            f20 = 1.0 - f18
        else:
            if f88 <= f90:
                f90 = f88 + 1
            else:
                f90 = f90 + 1
            f10 = f8
            f8 = 100.0 * c_arr[i]
            v8 = f8 - f10
            f28 = f20 * f28 + f18 * v8
            f30 = f18 * f28 + f20 * f30
            vC = 1.5 * f28 - 0.5 * f30
            f38 = f20 * f38 + f18 * vC
            f40 = f18 * f38 + f20 * f40
            v10 = 1.5 * f38 - 0.5 * f40
            f48 = f20 * f48 + f18 * v10
            f50 = f18 * f48 + f20 * f50
            v14 = 1.5 * f48 - 0.5 * f50
            f58 = f20 * f58 + f18 * abs(v8)
            f60 = f18 * f58 + f20 * f60
            v18 = 1.5 * f58 - 0.5 * f60
            f68 = f20 * f68 + f18 * v18
            f70 = f18 * f68 + f20 * f70
            v1C = 1.5 * f68 - 0.5 * f70
            f78 = f20 * f78 + f18 * v1C
            f80 = f18 * f78 + f20 * f80
            v20 = 1.5 * f78 - 0.5 * f80

            if f88 >= f90 and f8 != f10:
                f0 = 1.0
            if f88 == f90 and f0 == 0.0:
                f90 = 0.0

        if f88 < f90 and v20 > 0.0000000001:
            v4 = (v14 / v20 + 1.0) * 50.0
            if v4 > 100.0:
                v4 = 100.0
            if v4 < 0.0:
                v4 = 0.0
        else:
            v4 = 50.0
        result[i] = v4

    return result


# ---------------------------------------------------------------------------
# 2. JMA  (overlap/jma.py)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _jma_loop(
    c_arr: np.ndarray,
    m: int,
    sum_length: int,
    length: float,
    pr: float,
    length1: float,
    pow1: float,
    bet: float,
    beta: float,
    r_volty_max: float,
) -> np.ndarray:
    """Core JMA adaptive filter.  Returns jma array of length *m*."""
    jma = np.zeros(m)
    volty = np.zeros(m)
    v_sum = np.zeros(m)

    kv = 0.0
    det0 = 0.0
    det1 = 0.0
    ma2 = 0.0
    ma1 = c_arr[0]
    uBand = c_arr[0]
    lBand = c_arr[0]
    jma[0] = c_arr[0]

    avg_volty_sum = 0.0

    for i in range(1, m):
        price = c_arr[i]

        # Price volatility
        del1 = price - uBand
        del2 = price - lBand
        if abs(del1) != abs(del2):
            volty[i] = max(abs(del1), abs(del2))
        else:
            volty[i] = 0.0

        # Relative price volatility factor
        start = i - sum_length
        if start < 0:
            start = 0
        v_sum[i] = v_sum[i - 1] + (volty[i] - volty[start]) / sum_length

        # Incremental window sum over v_sum
        avg_volty_sum += v_sum[i]
        if i > 65:
            avg_volty_sum -= v_sum[i - 66]
        avg_volty_window = min(i + 1, 66)
        avg_volty = avg_volty_sum / avg_volty_window
        if avg_volty == 0:
            d_volty = 0.0
        else:
            d_volty = volty[i] / avg_volty
        r_volty = max(1.0, min(r_volty_max, d_volty))

        # Jurik volatility bands
        pow2 = r_volty**pow1
        kv = bet ** (pow2**0.5)
        if del1 > 0:
            uBand = price
        else:
            uBand = price - kv * del1
        if del2 < 0:
            lBand = price
        else:
            lBand = price - kv * del2

        # Jurik Dynamic Factor
        alpha = beta**pow2

        # 1st stage - preliminary smoothing by adaptive EMA
        ma1 = (1 - alpha) * price + alpha * ma1

        # 2nd stage - one more preliminary smoothing by Kalman filter
        det0 = (price - ma1) * (1 - beta) + beta * det0
        ma2 = ma1 + pr * det0

        # 3rd stage - final smoothing by unique Jurik adaptive filter
        det1 = (ma2 - jma[i - 1]) * (1 - alpha) * (1 - alpha) + alpha * alpha * det1
        jma[i] = jma[i - 1] + det1

    return jma


# ---------------------------------------------------------------------------
# 3. HWC  (volatility/hwc.py)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _hwc_loop(
    c_arr: np.ndarray,
    m: int,
    na: float,
    nb: float,
    nc: float,
    nd: float,
    scalar: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Core Holt-Winter Channel filter.

    Returns ``(result, upper, lower)`` arrays of length *m*.
    """
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

        # update state
        last_price = c_arr[i]
        last_a = A
        last_f = F
        last_v = V
        last_var = var
        last_result = result_arr[i]

    return result_arr, upper_arr, lower_arr


# ---------------------------------------------------------------------------
# 4. STC helper  (momentum/stc.py — schaff_tc inner loops)
#    The two loops look similar but differ in which array gates the condition:
#    loop 1 checks the *low* array, loop 2 checks the *range* array.
# ---------------------------------------------------------------------------
@njit(cache=True)
def _schaff_tc_loop(
    val_arr: np.ndarray,
    low_arr: np.ndarray,
    range_arr: np.ndarray,
    m: int,
    factor: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """First stochastic of MACD for Schaff Trend Cycle.

    Condition gates on ``low_arr[i] > 0``.
    Returns ``(stoch, smoothed)`` arrays of length *m*.
    """
    stoch = np.empty(m)
    smoothed = np.zeros(m)
    stoch[0] = 0.0

    for i in range(1, m):
        if low_arr[i] > 0:
            stoch[i] = 100.0 * ((val_arr[i] - low_arr[i]) / range_arr[i])
        else:
            stoch[i] = stoch[i - 1]
        smoothed[i] = smoothed[i - 1] + factor * (stoch[i] - smoothed[i - 1])

    return stoch, smoothed


@njit(cache=True)
def _schaff_tc_loop2(
    val_arr: np.ndarray,
    low_arr: np.ndarray,
    range_arr: np.ndarray,
    m: int,
    factor: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Second stochastic of smoothed PF for Schaff Trend Cycle.

    Condition gates on ``range_arr[i] > 0``.
    Returns ``(stoch, smoothed)`` arrays of length *m*.
    """
    stoch = np.empty(m)
    smoothed = np.zeros(m)
    stoch[0] = 0.0

    for i in range(1, m):
        if range_arr[i] > 0:
            stoch[i] = 100.0 * ((val_arr[i] - low_arr[i]) / range_arr[i])
        else:
            stoch[i] = stoch[i - 1]
        smoothed[i] = smoothed[i - 1] + factor * (stoch[i] - smoothed[i - 1])

    return stoch, smoothed


# ---------------------------------------------------------------------------
# 5. EBSW  (cycles/ebsw.py)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _ebsw_loop(
    c_arr: np.ndarray,
    m: int,
    length: int,
    alpha1: float,
    c1: float,
    c2: float,
    c3: float,
) -> np.ndarray:
    """Core Even Better SineWave filter.  Returns result array of length *m*."""
    result = np.full(m, np.nan)
    result[length - 1] = 0.0

    lastClose = 0.0
    lastHP = 0.0
    filt_p = 0.0  # one bar back
    filt_pp = 0.0  # two bars back

    for i in range(length, m):
        HP = 0.5 * (1 + alpha1) * (c_arr[i] - lastClose) + alpha1 * lastHP
        Filt = c1 * (HP + lastHP) / 2 + c2 * filt_p + c3 * filt_pp

        Wave = (Filt + filt_p + filt_pp) / 3
        Pwr = (Filt * Filt + filt_p * filt_p + filt_pp * filt_pp) / 3

        if Pwr > 0:
            Wave = Wave / (Pwr**0.5)
        else:
            Wave = 0.0

        filt_pp = filt_p
        filt_p = Filt
        lastHP = HP
        lastClose = c_arr[i]
        result[i] = Wave

    return result


# ---------------------------------------------------------------------------
# 6. QQE  (momentum/qqe.py)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _qqe_loop(
    rsi_arr: np.ndarray,
    ub_arr: np.ndarray,
    lb_arr: np.ndarray,
    m: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Core QQE iterative band/trend logic.

    Returns ``(long, short, trend, qqe, qqe_long, qqe_short)`` arrays.
    """
    long_arr = np.zeros(m)
    short_arr = np.zeros(m)
    trend_arr = np.ones(m)
    qqe_arr = np.empty(m)
    qqe_arr[0] = rsi_arr[0]
    qqe_long_arr = np.full(m, np.nan)
    qqe_short_arr = np.full(m, np.nan)

    for i in range(1, m):
        c_rsi = rsi_arr[i]
        p_rsi = rsi_arr[i - 1]
        c_long = long_arr[i - 1]
        c_short = short_arr[i - 1]
        if i >= 2:
            p_long = long_arr[i - 2]
            p_short = short_arr[i - 2]
        else:
            p_long = 0.0
            p_short = 0.0

        # Long Line
        if p_rsi > c_long and c_rsi > c_long:
            long_arr[i] = max(c_long, lb_arr[i])
        else:
            long_arr[i] = lb_arr[i]

        # Short Line
        if p_rsi < c_short and c_rsi < c_short:
            short_arr[i] = min(c_short, ub_arr[i])
        else:
            short_arr[i] = ub_arr[i]

        # Trend & QQE
        if (c_rsi > c_short and p_rsi < p_short) or (
            c_rsi <= c_short and p_rsi >= p_short
        ):
            trend_arr[i] = 1.0
            qqe_arr[i] = long_arr[i]
            qqe_long_arr[i] = long_arr[i]
        elif (c_rsi > c_long and p_rsi < p_long) or (
            c_rsi <= c_long and p_rsi >= p_long
        ):
            trend_arr[i] = -1.0
            qqe_arr[i] = short_arr[i]
            qqe_short_arr[i] = short_arr[i]
        else:
            trend_arr[i] = trend_arr[i - 1]
            if trend_arr[i] == 1.0:
                qqe_arr[i] = long_arr[i]
                qqe_long_arr[i] = long_arr[i]
            else:
                qqe_arr[i] = short_arr[i]
                qqe_short_arr[i] = short_arr[i]

    return long_arr, short_arr, trend_arr, qqe_arr, qqe_long_arr, qqe_short_arr


# ---------------------------------------------------------------------------
# 7. LRSI  (momentum/lrsi.py)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _lrsi_loop(
    c_arr: np.ndarray, n: int, gamma: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Laguerre filter.  Returns ``(l0, l1, l2, l3)`` arrays."""
    l0 = np.empty(n)
    l1 = np.empty(n)
    l2 = np.empty(n)
    l3 = np.empty(n)
    l0[0] = l1[0] = l2[0] = l3[0] = c_arr[0]

    for i in range(1, n):
        l0[i] = (1 - gamma) * c_arr[i] + gamma * l0[i - 1]
        l1[i] = -gamma * l0[i] + l0[i - 1] + gamma * l1[i - 1]
        l2[i] = -gamma * l1[i] + l1[i - 1] + gamma * l2[i - 1]
        l3[i] = -gamma * l2[i] + l2[i - 1] + gamma * l3[i - 1]

    return l0, l1, l2, l3


# ---------------------------------------------------------------------------
# 8. PMAX  (trend/pmax.py)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _pmax_loop(
    c_arr: np.ndarray,
    pmax_up_arr: np.ndarray,
    pmax_down_arr: np.ndarray,
    n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """PMAX band/trend loop.  Returns ``(trend, pmax)`` arrays."""
    trend_arr = np.ones(n)
    pmax_arr = np.zeros(n)

    for i in range(1, n):
        if c_arr[i - 1] > pmax_up_arr[i - 1]:
            pmax_up_arr[i] = max(pmax_up_arr[i], pmax_up_arr[i - 1])

        if c_arr[i - 1] < pmax_down_arr[i - 1]:
            pmax_down_arr[i] = min(pmax_down_arr[i], pmax_down_arr[i - 1])

        if c_arr[i] > pmax_down_arr[i - 1]:
            trend_arr[i] = 1.0
        elif c_arr[i] < pmax_up_arr[i - 1]:
            trend_arr[i] = -1.0
        else:
            trend_arr[i] = trend_arr[i - 1]

        if trend_arr[i] == 1.0:
            pmax_arr[i] = pmax_up_arr[i]
        else:
            pmax_arr[i] = pmax_down_arr[i]

    return trend_arr, pmax_arr


# ---------------------------------------------------------------------------
# 9. Fisher Transform  (momentum/fisher.py)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _fisher_loop(pos_arr: np.ndarray, m: int, length: int) -> np.ndarray:
    """Fisher Transform iterative loop.  Returns result array."""
    result = np.full(m, np.nan)
    result[length - 1] = 0.0

    v = 0.0
    for i in range(length, m):
        v = 0.66 * pos_arr[i] + 0.67 * v
        if v < -0.99:
            v = -0.999
        elif v > 0.99:
            v = 0.999
        result[i] = 0.5 * (np.log((1 + v) / (1 - v)) + result[i - 1])

    return result


# ---------------------------------------------------------------------------
# 10. PSAR  (trend/psar.py)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _psar_loop(
    h_arr: np.ndarray,
    l_arr: np.ndarray,
    m: int,
    falling: bool,
    sar: float,
    ep: float,
    af0: float,
    max_af: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parabolic SAR iterative loop.

    Returns ``(long, short, af, reversal)`` arrays.
    """
    long_arr = np.full(m, np.nan)
    short_arr = np.full(m, np.nan)
    af_arr = np.full(m, np.nan)
    reversal_arr = np.zeros(m)

    af_arr[0] = af0
    if m > 1:
        af_arr[1] = af0
    af = af0

    for row in range(1, m):
        high_ = h_arr[row]
        low_ = l_arr[row]

        if falling:
            _sar = sar + af * (ep - sar)
            reverse = high_ > _sar

            if low_ < ep:
                ep = low_
                af = min(af + af0, max_af)

            prev1 = h_arr[row - 1]
            prev2 = h_arr[row - 2] if row >= 2 else prev1
            _sar = max(prev1, prev2, _sar)
        else:
            _sar = sar + af * (ep - sar)
            reverse = low_ < _sar

            if high_ > ep:
                ep = high_
                af = min(af + af0, max_af)

            prev1 = l_arr[row - 1]
            prev2 = l_arr[row - 2] if row >= 2 else prev1
            _sar = min(prev1, prev2, _sar)

        if reverse:
            _sar = ep
            af = af0
            falling = not falling
            ep = low_ if falling else high_

        sar = _sar

        if falling:
            short_arr[row] = sar
        else:
            long_arr[row] = sar

        af_arr[row] = af
        reversal_arr[row] = 1.0 if reverse else 0.0

    return long_arr, short_arr, af_arr, reversal_arr


# ---------------------------------------------------------------------------
# 11. MCGD  (overlap/mcgd.py)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _mcgd_loop(c_arr: np.ndarray, n: int, c: float, length: int) -> np.ndarray:
    """McGinley Dynamic iterative loop.  Returns result array."""
    result = np.empty(n)
    result[0] = c_arr[0]

    for i in range(1, n):
        if result[i - 1] != 0:
            denom = c * length * (c_arr[i] / result[i - 1]) ** 4
            if denom < 1e-10:
                denom = 1e-10
            result[i] = result[i - 1] + (c_arr[i] - result[i - 1]) / denom
        else:
            result[i] = c_arr[i]

    return result


# ---------------------------------------------------------------------------
# 12. HWMA  (overlap/hwma.py)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _hwma_loop(
    c_arr: np.ndarray, m: int, na: float, nb: float, nc: float
) -> np.ndarray:
    """Holt-Winter Moving Average loop.  Returns result array."""
    result = np.empty(m)
    last_a = 0.0
    last_v = 0.0
    last_f = c_arr[0]

    for i in range(m):
        F = (1.0 - na) * (last_f + last_v + 0.5 * last_a) + na * c_arr[i]
        V = (1.0 - nb) * (last_v + last_a) + nb * (F - last_f)
        A = (1.0 - nc) * last_a + nc * (V - last_v)
        result[i] = F + V + 0.5 * A
        last_a = A
        last_f = F
        last_v = V

    return result


# ---------------------------------------------------------------------------
# 13. SuperTrend  (overlap/supertrend.py)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _supertrend_loop(
    c_arr: np.ndarray,
    ub_arr: np.ndarray,
    lb_arr: np.ndarray,
    m: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """SuperTrend band/direction loop.

    Returns ``(dir_, trend, long, short)`` arrays.
    Note: *ub_arr* and *lb_arr* are modified in-place.
    """
    dir_ = np.ones(m)
    trend = np.zeros(m)
    long = np.full(m, np.nan)
    short = np.full(m, np.nan)

    for i in range(1, m):
        if c_arr[i] > ub_arr[i - 1]:
            dir_[i] = 1.0
        elif c_arr[i] < lb_arr[i - 1]:
            dir_[i] = -1.0
        else:
            dir_[i] = dir_[i - 1]
            if dir_[i] > 0 and lb_arr[i] < lb_arr[i - 1]:
                lb_arr[i] = lb_arr[i - 1]
            if dir_[i] < 0 and ub_arr[i] > ub_arr[i - 1]:
                ub_arr[i] = ub_arr[i - 1]

        if dir_[i] > 0:
            trend[i] = lb_arr[i]
            long[i] = lb_arr[i]
        else:
            trend[i] = ub_arr[i]
            short[i] = ub_arr[i]

    return dir_, trend, long, short


# ---------------------------------------------------------------------------
# 14. VIDYA  (overlap/vidya.py)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _vidya_loop(
    c_arr: np.ndarray,
    cmo_arr: np.ndarray,
    m: int,
    length: int,
    alpha: float,
    seed: float,
) -> np.ndarray:
    """VIDYA adaptive EMA loop.  Returns result array."""
    result = np.full(m, np.nan)
    result[length - 1] = seed

    for i in range(length, m):
        result[i] = alpha * cmo_arr[i] * c_arr[i] + result[i - 1] * (
            1 - alpha * cmo_arr[i]
        )

    return result


# ---------------------------------------------------------------------------
# 15. SSF  (overlap/ssf.py)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _ssf2_loop(
    c_arr: np.ndarray, ssf_arr: np.ndarray, m: int, c1: float, b1: float, a1: float
) -> np.ndarray:
    """Super Smoother Filter (2-pole) loop.  Modifies *ssf_arr* in-place."""
    for i in range(2, m):
        ssf_arr[i] = c1 * c_arr[i] + b1 * ssf_arr[i - 1] + a1 * ssf_arr[i - 2]
    return ssf_arr


@njit(cache=True)
def _ssf3_loop(
    c_arr: np.ndarray,
    ssf_arr: np.ndarray,
    m: int,
    c1: float,
    c2: float,
    c3: float,
    c4: float,
) -> np.ndarray:
    """Super Smoother Filter (3-pole) loop.  Modifies *ssf_arr* in-place."""
    for i in range(3, m):
        ssf_arr[i] = (
            c1 * c_arr[i]
            + c2 * ssf_arr[i - 1]
            + c3 * ssf_arr[i - 2]
            + c4 * ssf_arr[i - 3]
        )
    return ssf_arr


# ---------------------------------------------------------------------------
# 16. Hilbert Transform  (cycles/_hilbert.py)
# ---------------------------------------------------------------------------
@njit(cache=True)
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

    # State for trend mode detection (matches TA-Lib)
    days_in_trend = 0
    prev_sine = 0.0
    prev_lead_sine = 0.0
    prev_dc_phase = 0.0

    # TA-Lib warmup: WMA-only warmup before the Hilbert starts.
    # The Hilbert computation begins at bar ht_start with empty circular
    # buffers.  smooth_price[0..ht_start-1] stays 0 so FIR taps that
    # reference bars before ht_start read 0, replicating the empty-buffer
    # startup.
    #   ht_start = 12  →  HT_DCPERIOD, HT_PHASOR  (lookback 32)
    #   ht_start = 37  →  HT_DCPHASE, HT_SINE, HT_TRENDMODE,
    #                      HT_TRENDLINE  (lookback 63)

    for i in range(m):
        # WMA(4) smoothing — only stored from bar ht_start onwards
        # (matching TA-Lib which stores smoothPrice only in the main loop)
        if i >= ht_start and i >= 3:
            smooth_price[i] = (
                4.0 * close_arr[i]
                + 3.0 * close_arr[i - 1]
                + 2.0 * close_arr[i - 2]
                + close_arr[i - 3]
            ) / 10.0
        # smooth_price[0..ht_start-1] stays 0 (from np.zeros init)

        if i < ht_start:
            period[i] = 0.0
            smooth_period_arr[i] = 0.0
            continue

        adj = 0.075 * period[i - 1] + 0.54

        # Detrend (4-tap FIR — equivalent to TA-Lib's even/odd macro)
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
                angle = 2.0 * np.pi * j / dc_period_int
                real_part += np.sin(angle) * smooth_price[i - j]
                imag_part += np.cos(angle) * smooth_price[i - j]

        # TA-Lib DCPhase: fabs(imagPart) > 0 → atan; else adjust previous
        abs_imag = abs(imag_part)
        if abs_imag > 0.0:
            dc_phase_val = np.arctan(real_part / imag_part) * 180.0 / np.pi
        else:
            # Adjust previous DCPhase incrementally (TA-Lib behaviour)
            dc_phase_val = prev_dc_phase
            if real_part < 0.0:
                dc_phase_val -= 90.0
            elif real_part > 0.0:
                dc_phase_val += 90.0

        dc_phase_val += 90.0
        # Compensate for one bar lag of the weighted moving average
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

        # Instantaneous Trendline (ITrend) — computed BEFORE trend mode
        # because trend mode uses trendline for price-divergence check.
        dc_per = max(int(sp + 0.5), 1)
        itrend_sum = 0.0
        for j in range(dc_per):
            if i - j >= 0:
                itrend_sum += close_arr[i - j]
        it_trend[i] = itrend_sum / dc_per
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

        # Step 2: Not enough bars since crossover → cycle mode
        if days_in_trend < 0.5 * sp:
            trend = 0

        # Step 3: Phase change in expected cycle range → cycle mode
        phase_diff = dc_phase_val - prev_dc_phase
        if (
            sp != 0.0
            and phase_diff > 0.67 * 360.0 / sp
            and phase_diff < 1.5 * 360.0 / sp
        ):
            trend = 0

        # Step 4: Price far from trendline → trend mode override
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


# ---------------------------------------------------------------------------
# 17. SAREXT  (trend/sarext.py)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _sarext_loop(
    h_arr: np.ndarray,
    l_arr: np.ndarray,
    m: int,
    is_long: bool,
    sar: float,
    ep: float,
    af_init_long: float,
    af_long: float,
    af_max_long: float,
    af_init_short: float,
    af_short: float,
    af_max_short: float,
    offset_on_reverse: float,
) -> np.ndarray:
    """Extended Parabolic SAR loop (TA-Lib compatible).

    Faithfully reproduces TA-Lib's SAREXT main loop.  Key details:
    - Bar 0 = NaN  (lookback = 1)
    - The loop starts at bar 1 (todayIdx = startIdx = 1) and may
      reverse on the very first bar.
    - ``newLow``/``newHigh`` are initialised to bar 1 data (TA-Lib
      overwrites them after the SAR/EP init), so the first iteration's
      ``prev`` and ``new`` both refer to bar 1.
    - ``offsetOnReverse`` is multiplicative.
    - After reversal the SAR is stepped once and clamped.

    Returns a single array: positive = long SAR, negative = short SAR.
    """
    result = np.full(m, np.nan)
    if m < 2:
        return result

    af_l = af_init_long
    af_s = af_init_short

    # TA-Lib overwrites newLow/newHigh to bar 1 data BEFORE the loop.
    # The loop's first iteration reads bar 1 again, so prev == new on
    # the first pass — exactly matching TA-Lib.
    new_high = h_arr[1]
    new_low = l_arr[1]

    for row in range(1, m):
        prev_low = new_low
        prev_high = new_high
        new_low = l_arr[row]
        new_high = h_arr[row]

        if is_long:
            if new_low <= sar:
                # Long -> Short reversal
                is_long = False
                sar = ep
                # Clamp overridden SAR within prev/current range
                if sar < prev_high:
                    sar = prev_high
                if sar < new_high:
                    sar = new_high
                # Multiplicative offset
                if offset_on_reverse != 0.0:
                    sar += sar * offset_on_reverse
                result[row] = -sar

                af_s = af_init_short
                ep = new_low
                # Post-reversal step + clamp
                sar = sar + af_s * (ep - sar)
                if sar < prev_high:
                    sar = prev_high
                if sar < new_high:
                    sar = new_high
            else:
                # No reversal
                result[row] = sar
                if new_high > ep:
                    ep = new_high
                    af_l += af_long
                    if af_l > af_max_long:
                        af_l = af_max_long
                sar = sar + af_l * (ep - sar)
                if sar > prev_low:
                    sar = prev_low
                if sar > new_low:
                    sar = new_low
        else:
            if new_high >= sar:
                # Short -> Long reversal
                is_long = True
                sar = ep
                # Clamp
                if sar > prev_low:
                    sar = prev_low
                if sar > new_low:
                    sar = new_low
                # Multiplicative offset
                if offset_on_reverse != 0.0:
                    sar -= sar * offset_on_reverse
                result[row] = sar

                af_l = af_init_long
                ep = new_high
                # Post-reversal step + clamp
                sar = sar + af_l * (ep - sar)
                if sar > prev_low:
                    sar = prev_low
                if sar > new_low:
                    sar = new_low
            else:
                # No reversal
                result[row] = -sar
                if new_low < ep:
                    ep = new_low
                    af_s += af_short
                    if af_s > af_max_short:
                        af_s = af_max_short
                sar = sar + af_s * (ep - sar)
                if sar < prev_high:
                    sar = prev_high
                if sar < new_high:
                    sar = new_high

    return result


__all__ = [
    "_rsx_loop",
    "_jma_loop",
    "_hwc_loop",
    "_schaff_tc_loop",
    "_schaff_tc_loop2",
    "_ebsw_loop",
    "_qqe_loop",
    "_lrsi_loop",
    "_pmax_loop",
    "_fisher_loop",
    "_psar_loop",
    "_mcgd_loop",
    "_hwma_loop",
    "_supertrend_loop",
    "_vidya_loop",
    "_ssf2_loop",
    "_ssf3_loop",
    "_hilbert_transform_loop",
    "_sarext_loop",
]
