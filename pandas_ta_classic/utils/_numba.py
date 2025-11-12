# -*- coding: utf-8 -*-
"""
Numba-accelerated helper functions for pandas-ta-classic

This module contains performance-critical calculations optimized with Numba JIT compilation.
All functions gracefully fall back to pure Python/NumPy implementations if Numba is not available.

Requirements:
    - numba>=0.58.0 (optional)

Performance Notes:
    - Numba compilation happens on first call (warmup overhead)
    - Subsequent calls benefit from compiled machine code
    - Best performance gains on large datasets (1000+ rows)
"""

import numpy as np
from pandas_ta_classic import Imports

# Try to import numba, gracefully handle if not available
if Imports["numba"]:
    try:
        from numba import jit

        NUMBA_AVAILABLE = True
    except ImportError:
        NUMBA_AVAILABLE = False

        # Create a no-op decorator for compatibility
        def jit(*args, **kwargs):
            def decorator(func):
                return func

            return decorator

else:
    NUMBA_AVAILABLE = False

    # Create a no-op decorator for compatibility
    def jit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


@jit(nopython=True, cache=True)
def rsx_numba_core(close_values, length):
    """
    Numba-optimized core calculation for RSX (Relative Strength Xtra) indicator.

    This function implements the Jurik RSX algorithm using loops that benefit
    significantly from JIT compilation.

    Args:
        close_values (np.ndarray): Array of close prices
        length (int): RSX period length

    Returns:
        np.ndarray: RSX values

    Performance:
        - ~10-50x faster than pure Python loops on large datasets
        - Compilation overhead on first call (~1-2 seconds)
    """
    m = len(close_values)
    result = np.full(m, np.nan)

    # Initialize result with NaN for warmup period and 0 for first valid value
    for i in range(length - 1):
        result[i] = np.nan
    if length - 1 < m:
        result[length - 1] = 0.0

    # Variables
    vC, v1C = 0.0, 0.0
    v4, v8, v10, v14, v18, v20 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    f0, f8, f10, f18, f20, f28, f30, f38 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    f40, f48, f50, f58, f60, f68, f70, f78 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    f80, f88, f90 = 0.0, 0.0, 0.0

    for i in range(length, m):
        if f90 == 0.0:
            f90 = 1.0
            f0 = 0.0
            if length - 1.0 >= 5:
                f88 = length - 1.0
            else:
                f88 = 5.0
            f8 = 100.0 * close_values[i]
            f18 = 3.0 / (length + 2.0)
            f20 = 1.0 - f18
        else:
            if f88 <= f90:
                f90 = f88 + 1.0
            else:
                f90 = f90 + 1.0
            f10 = f8
            f8 = 100.0 * close_values[i]
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


@jit(nopython=True, cache=True)
def supertrend_numba_core(close_values, upperband_values, lowerband_values):
    """
    Numba-optimized core calculation for Supertrend indicator.

    Args:
        close_values (np.ndarray): Close prices
        upperband_values (np.ndarray): Upper band values (hl2 + multiplier * ATR)
        lowerband_values (np.ndarray): Lower band values (hl2 - multiplier * ATR)

    Returns:
        tuple: (long, short, direction, trend) arrays

    Performance:
        - ~5-20x faster than pure Python loops
    """
    m = len(close_values)
    long = np.full(m, np.nan)
    short = np.full(m, np.nan)
    direction = np.ones(m)
    trend = np.zeros(m)

    # Initialize upperband and lowerband as mutable copies
    upperband = upperband_values.copy()
    lowerband = lowerband_values.copy()

    for i in range(1, m):
        # Determine direction based on close vs bands
        if close_values[i] > upperband[i - 1]:
            direction[i] = 1.0
        elif close_values[i] < lowerband[i - 1]:
            direction[i] = -1.0
        else:
            direction[i] = direction[i - 1]
            # Adjust bands if continuing in same direction
            if direction[i] > 0 and lowerband[i] < lowerband[i - 1]:
                lowerband[i] = lowerband[i - 1]
            if direction[i] < 0 and upperband[i] > upperband[i - 1]:
                upperband[i] = upperband[i - 1]

        # Set long/short/trend values
        if direction[i] > 0:
            trend[i] = long[i] = lowerband[i]
        else:
            trend[i] = short[i] = upperband[i]

    return long, short, direction, trend


@jit(nopython=True, cache=True)
def fisher_numba_core(position_values, length):
    """
    Numba-optimized core calculation for Fisher Transform indicator.

    Args:
        position_values (np.ndarray): Normalized position values
        length (int): Fisher period length

    Returns:
        np.ndarray: Fisher transform values

    Performance:
        - ~20-100x faster than pure Python loops
    """
    m = len(position_values)
    result = np.full(m, np.nan)

    # Initialize warmup period
    for i in range(length - 1):
        result[i] = np.nan
    if length - 1 < m:
        result[length - 1] = 0.0

    v = 0.0
    for i in range(length, m):
        v = 0.66 * position_values[i] + 0.67 * v
        if v < -0.99:
            v = -0.999
        if v > 0.99:
            v = 0.999
        result[i] = 0.5 * (np.log((1 + v) / (1 - v)) + result[i - 1])

    return result


@jit(nopython=True, cache=True)
def qqe_numba_core(rsi_ma_values, upperband_values, lowerband_values):
    """
    Numba-optimized core calculation for QQE indicator.

    Args:
        rsi_ma_values (np.ndarray): RSI moving average values
        upperband_values (np.ndarray): Upper band values
        lowerband_values (np.ndarray): Lower band values

    Returns:
        tuple: (long, short, trend, qqe, qqe_long, qqe_short) arrays

    Performance:
        - ~15-50x faster than pure Python loops
    """
    m = len(rsi_ma_values)
    long = np.full(m, 0.0)
    short = np.full(m, 0.0)
    trend = np.full(m, 1.0)
    qqe = np.full(m, np.nan)
    qqe_long = np.full(m, np.nan)
    qqe_short = np.full(m, np.nan)

    qqe[0] = rsi_ma_values[0]

    for i in range(1, m):
        c_rsi = rsi_ma_values[i]
        p_rsi = rsi_ma_values[i - 1]
        c_long = long[i - 1]
        c_short = short[i - 1]

        if i >= 2:
            p_long = long[i - 2]
            p_short = short[i - 2]
        else:
            p_long = 0.0
            p_short = 0.0

        # Long Line
        if p_rsi > c_long and c_rsi > c_long:
            long[i] = max(c_long, lowerband_values[i])
        else:
            long[i] = lowerband_values[i]

        # Short Line
        if p_rsi < c_short and c_rsi < c_short:
            short[i] = min(c_short, upperband_values[i])
        else:
            short[i] = upperband_values[i]

        # Trend & QQE Calculation
        if (c_rsi > c_short and p_rsi < p_short) or (
            c_rsi <= c_short and p_rsi >= p_short
        ):
            trend[i] = 1.0
            qqe[i] = qqe_long[i] = long[i]
        elif (c_rsi > c_long and p_rsi < p_long) or (
            c_rsi <= c_long and p_rsi >= p_long
        ):
            trend[i] = -1.0
            qqe[i] = qqe_short[i] = short[i]
        else:
            trend[i] = trend[i - 1]
            if trend[i] == 1.0:
                qqe[i] = qqe_long[i] = long[i]
            else:
                qqe[i] = qqe_short[i] = short[i]

    return long, short, trend, qqe, qqe_long, qqe_short


@jit(nopython=True, cache=True)
def psar_numba_core(high_values, low_values, af0, af_step, max_af):
    """
    Numba-optimized core calculation for Parabolic SAR indicator.

    Args:
        high_values (np.ndarray): High prices
        low_values (np.ndarray): Low prices
        af0 (float): Initial acceleration factor
        af_step (float): AF increment step
        max_af (float): Maximum acceleration factor

    Returns:
        tuple: (psar, long, short, af, reversal) arrays

    Performance:
        - ~10-40x faster than pure Python loops
    """
    m = len(high_values)
    psar = np.full(m, np.nan)
    long = np.full(m, np.nan)
    short = np.full(m, np.nan)
    af = np.full(m, np.nan)
    reversal = np.zeros(m)

    # Determine initial direction
    if low_values[1] - low_values[0] > high_values[0] - high_values[1]:
        falling = False
        sar = low_values[0]
        ep = high_values[0]
    else:
        falling = True
        sar = high_values[0]
        ep = low_values[0]

    psar[0] = sar
    af[0] = af0
    af[1] = af0
    current_af = af0

    for i in range(1, m):
        high_ = high_values[i]
        low_ = low_values[i]

        if falling:
            _sar = sar + current_af * (ep - sar)
            reverse = high_ > _sar

            if low_ < ep:
                ep = low_
                current_af = min(current_af + af_step, max_af)

            if i >= 2:
                _sar = max(high_values[i - 1], high_values[i - 2], _sar)
            else:
                _sar = max(high_values[i - 1], _sar)
        else:
            _sar = sar + current_af * (ep - sar)
            reverse = low_ < _sar

            if high_ > ep:
                ep = high_
                current_af = min(current_af + af_step, max_af)

            if i >= 2:
                _sar = min(low_values[i - 1], low_values[i - 2], _sar)
            else:
                _sar = min(low_values[i - 1], _sar)

        if reverse:
            _sar = ep
            current_af = af0
            falling = not falling
            ep = low_ if falling else high_
            reversal[i] = 1.0

        sar = _sar
        psar[i] = sar
        af[i] = current_af

        if falling:
            short[i] = sar
        else:
            long[i] = sar

    return psar, long, short, af, reversal


@jit(nopython=True, cache=True)
def stc_numba_core(xmacd_values, tclength, factor):
    """
    Numba-optimized core calculation for Schaff Trend Cycle (STC) indicator.

    Args:
        xmacd_values (np.ndarray): MACD or oscillator values
        tclength (int): TC length
        factor (float): Smoothing factor

    Returns:
        tuple: (pff, pf) arrays - final STC and intermediate stochastic

    Performance:
        - ~20-80x faster than pure Python loops
    """
    m = len(xmacd_values)

    # First stochastic of MACD
    stoch1 = np.zeros(m)
    pf = np.zeros(m)

    for i in range(1, m):
        # Calculate rolling min/max for xmacd
        start_idx = max(0, i - tclength + 1)
        window = xmacd_values[start_idx : i + 1]
        lowest_xmacd = np.min(window)
        highest_xmacd = np.max(window)
        xmacd_range = highest_xmacd - lowest_xmacd

        if xmacd_range > 0.0:
            stoch1[i] = 100.0 * ((xmacd_values[i] - lowest_xmacd) / xmacd_range)
        else:
            stoch1[i] = stoch1[i - 1]

        # Smoothed calculation for % Fast D of MACD
        pf[i] = pf[i - 1] + (factor * (stoch1[i] - pf[i - 1]))

    # Second stochastic of smoothed PF
    stoch2 = np.zeros(m)
    pff = np.zeros(m)

    for i in range(1, m):
        # Calculate rolling min/max for pf
        start_idx = max(0, i - tclength + 1)
        window = pf[start_idx : i + 1]
        lowest_pf = np.min(window)
        highest_pf = np.max(window)
        pf_range = highest_pf - lowest_pf

        if pf_range > 0.0:
            stoch2[i] = 100.0 * ((pf[i] - lowest_pf) / pf_range)
        else:
            stoch2[i] = stoch2[i - 1]

        # Smoothed calculation for % Fast D of PF
        pff[i] = pff[i - 1] + (factor * (stoch2[i] - pff[i - 1]))

    return pff, pf


def get_numba_status():
    """
    Get the current status of Numba availability and configuration.

    Returns:
        dict: Status information including availability, version, and threading
    """
    status = {
        "available": NUMBA_AVAILABLE,
        "version": None,
        "threading_layer": None,
        "num_threads": None,
    }

    if NUMBA_AVAILABLE:
        try:
            import numba

            status["version"] = numba.__version__

            # Try to get threading configuration
            try:
                from numba import config

                status["threading_layer"] = config.THREADING_LAYER
                status["num_threads"] = config.NUMBA_NUM_THREADS
            except (ImportError, AttributeError):
                pass
        except ImportError:
            status["available"] = False

    return status


def print_numba_info():
    """Print detailed information about Numba configuration."""
    status = get_numba_status()

    print("=" * 60)
    print("Numba Configuration for pandas-ta-classic")
    print("=" * 60)
    print(f"Numba Available: {status['available']}")

    if status["available"]:
        print(f"Numba Version: {status['version']}")
        if status["threading_layer"]:
            print(f"Threading Layer: {status['threading_layer']}")
        if status["num_threads"]:
            print(f"Number of Threads: {status['num_threads']}")
        print("\nPerformance optimizations are ENABLED")
        print("JIT-compiled functions will be used for better performance.")
    else:
        print("\nPerformance optimizations are DISABLED")
        print("Install numba for better performance: pip install numba")
        print("Functions will fall back to pure Python/NumPy implementations.")
    print("=" * 60)


@jit(nopython=True, cache=True)
def kama_numba_core(close_values, sc_values, length):
    """
    Numba-optimized core calculation for KAMA (Kaufman Adaptive Moving Average).

    Args:
        close_values: Array of close prices
        sc_values: Array of pre-calculated smoothing constants (squared ER * (fr-sr) + sr)
        length: The lookback period

    Returns:
        Array of KAMA values
    """
    m = len(close_values)
    result = np.empty(m, dtype=np.float64)
    result[:length] = np.nan

    # Initialize first KAMA value
    result[length - 1] = 0.0

    # Calculate KAMA iteratively
    for i in range(length, m):
        result[i] = sc_values[i] * close_values[i] + (1 - sc_values[i]) * result[i - 1]

    return result


@jit(nopython=True, cache=True)
def vidya_numba_core(close_values, abs_cmo_values, alpha, length):
    """
    Numba-optimized core calculation for VIDYA (Variable Index Dynamic Average).

    Args:
        close_values: Array of close prices
        abs_cmo_values: Array of absolute CMO values
        alpha: Alpha value (2 / (length + 1))
        length: The lookback period

    Returns:
        Array of VIDYA values
    """
    m = len(close_values)
    result = np.empty(m, dtype=np.float64)
    result[:length] = np.nan

    # Initialize first VIDYA value
    result[length - 1] = 0.0

    # Calculate VIDYA iteratively
    for i in range(length, m):
        result[i] = alpha * abs_cmo_values[i] * close_values[i] + result[i - 1] * (
            1 - alpha * abs_cmo_values[i]
        )

    return result


@jit(nopython=True, cache=True)
def alma_numba_core(close_values, weights, length):
    """
    Numba-optimized core calculation for ALMA (Arnaud Legoux Moving Average).

    Args:
        close_values: Array of close prices
        weights: Pre-calculated Gaussian weights array
        length: Window length

    Returns:
        Array of ALMA values
    """
    m = len(close_values)
    result = np.empty(m, dtype=np.float64)
    result[: length - 1] = np.nan
    result[length - 1] = 0.0  # Match Python version behavior
    result[length] = np.nan  # Python version appends NaN at position `length`

    # Calculate ALMA for each position (start at length+1)
    for i in range(length + 1, m):
        window_sum = 0.0
        cum_sum = 0.0

        for j in range(length):
            window_sum += weights[j] * close_values[i - j]
            cum_sum += weights[j]

        result[i] = window_sum / cum_sum

    return result


@jit(nopython=True, cache=True)
def ssf_numba_core(close_values, c1, c2, c3, c4, poles):
    """
    Numba-optimized core calculation for SSF (Super Smoother Filter).

    Args:
        close_values: Array of close prices
        c1, c2, c3, c4: Pre-calculated filter coefficients
        poles: Number of poles (2 or 3)

    Returns:
        Array of SSF values
    """
    m = len(close_values)
    result = close_values.copy()  # Start with a copy to match Python .copy() behavior

    if poles == 3:
        # 3-pole filter - Python accesses negative indices which wrap around
        for i in range(m):
            if i == 0:
                # When i=0: ssf.iloc[-1], ssf.iloc[-2], ssf.iloc[-3] access last 3 elements
                result[i] = (
                    c1 * close_values[i]
                    + c2 * result[m - 1]  # result[-1]
                    + c3 * result[m - 2]  # result[-2]
                    + c4 * result[m - 3]  # result[-3]
                )
            elif i == 1:
                result[i] = (
                    c1 * close_values[i]
                    + c2 * result[0]  # result[i-1]
                    + c3 * result[m - 1]  # result[-1]
                    + c4 * result[m - 2]  # result[-2]
                )
            elif i == 2:
                result[i] = (
                    c1 * close_values[i]
                    + c2 * result[1]  # result[i-1]
                    + c3 * result[0]  # result[i-2]
                    + c4 * result[m - 1]  # result[-1]
                )
            else:
                result[i] = (
                    c1 * close_values[i]
                    + c2 * result[i - 1]
                    + c3 * result[i - 2]
                    + c4 * result[i - 3]
                )
    else:
        # 2-pole filter
        for i in range(m):
            if i == 0:
                # When i=0: ssf.iloc[-1], ssf.iloc[-2] access last 2 elements
                result[i] = (
                    c1 * close_values[i] + c2 * result[m - 1] + c3 * result[m - 2]
                )
            elif i == 1:
                result[i] = c1 * close_values[i] + c2 * result[0] + c3 * result[m - 1]
            else:
                result[i] = (
                    c1 * close_values[i] + c2 * result[i - 1] + c3 * result[i - 2]
                )

    return result


@jit(nopython=True, cache=True)
def hilo_numba_core(close_values, high_ma_values, low_ma_values):
    """
    Numba-optimized core calculation for HiLo (Gann HiLo Activator).

    Args:
        close_values: Array of close prices
        high_ma_values: Array of high moving average values
        low_ma_values: Array of low moving average values

    Returns:
        Tuple of (hilo, long, short) arrays
    """
    m = len(close_values)
    hilo = np.empty(m, dtype=np.float64)
    long = np.empty(m, dtype=np.float64)
    short = np.empty(m, dtype=np.float64)

    hilo[0] = np.nan
    long[0] = np.nan
    short[0] = np.nan

    # Calculate HiLo iteratively
    for i in range(1, m):
        if close_values[i] > high_ma_values[i - 1]:
            hilo[i] = low_ma_values[i]
            long[i] = low_ma_values[i]
            short[i] = np.nan
        elif close_values[i] < low_ma_values[i - 1]:
            hilo[i] = high_ma_values[i]
            long[i] = np.nan
            short[i] = high_ma_values[i]
        else:
            hilo[i] = hilo[i - 1]
            long[i] = hilo[i - 1]
            short[i] = hilo[i - 1]

    return hilo, long, short


@jit(nopython=True, cache=True)
def jma_numba_core(close_values, length, phase):
    """
    Numba-optimized core calculation for JMA (Jurik Moving Average).

    This is a complex adaptive moving average with multiple smoothing stages
    and dynamic volatility-based adaptation.

    Args:
        close_values: Array of close prices
        length: Smoothing period (internally adjusted)
        phase: Phase parameter (-100 to 100)

    Returns:
        Array of JMA values
    """
    m = len(close_values)
    result = np.empty(m, dtype=np.float64)
    volty = np.zeros(m, dtype=np.float64)
    v_sum = np.zeros(m, dtype=np.float64)

    # Static variables
    sum_length = 10
    _length = 0.5 * (length - 1)

    # Phase calculation
    if phase < -100:
        pr = 0.5
    elif phase > 100:
        pr = 2.5
    else:
        pr = 1.5 + phase * 0.01

    # Length calculations
    length1 = max(np.log(np.sqrt(_length)) / np.log(2.0) + 2.0, 0.0)
    pow1 = max(length1 - 2.0, 0.5)
    length2 = length1 * np.sqrt(_length)
    bet = length2 / (length2 + 1)
    beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2.0)

    # Initialize state variables
    kv = det0 = det1 = ma2 = 0.0
    result[0] = ma1 = uBand = lBand = close_values[0]

    # Main calculation loop
    for i in range(1, m):
        price = close_values[i]

        # Price volatility
        del1 = price - uBand
        del2 = price - lBand
        volty[i] = max(abs(del1), abs(del2)) if abs(del1) != abs(del2) else 0.0

        # Relative price volatility factor
        v_sum[i] = (
            v_sum[i - 1] + (volty[i] - volty[max(i - sum_length, 0)]) / sum_length
        )

        # Average volatility calculation
        start_idx = max(i - 65, 0)
        avg_volty = np.mean(v_sum[start_idx : i + 1])
        d_volty = 0.0 if avg_volty == 0.0 else volty[i] / avg_volty
        r_volty = max(1.0, min(np.power(length1, 1 / pow1), d_volty))

        # Jurik volatility bands
        pow2 = np.power(r_volty, pow1)
        kv = np.power(bet, np.sqrt(pow2))
        uBand = price if (del1 > 0) else price - (kv * del1)
        lBand = price if (del2 < 0) else price - (kv * del2)

        # Jurik Dynamic Factor
        power = np.power(r_volty, pow1)
        alpha = np.power(beta, power)

        # 1st stage - preliminary smoothing by adaptive EMA
        ma1 = ((1 - alpha) * price) + (alpha * ma1)

        # 2nd stage - one more preliminary smoothing by Kalman filter
        det0 = ((price - ma1) * (1 - beta)) + (beta * det0)
        ma2 = ma1 + pr * det0

        # 3rd stage - final smoothing by unique Jurik adaptive filter
        det1 = ((ma2 - result[i - 1]) * (1 - alpha) * (1 - alpha)) + (
            alpha * alpha * det1
        )
        result[i] = result[i - 1] + det1

    return result


# Module-level exports
__all__ = [
    "NUMBA_AVAILABLE",
    "rsx_numba_core",
    "supertrend_numba_core",
    "fisher_numba_core",
    "qqe_numba_core",
    "psar_numba_core",
    "stc_numba_core",
    "kama_numba_core",
    "vidya_numba_core",
    "alma_numba_core",
    "ssf_numba_core",
    "hilo_numba_core",
    "jma_numba_core",
    "get_numba_status",
    "print_numba_info",
]
