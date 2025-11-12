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
def supertrend_numba_core(hl2_values, atr_values, length, multiplier):
    """
    Numba-optimized core calculation for Supertrend indicator.
    
    Args:
        hl2_values (np.ndarray): (High + Low) / 2 values
        atr_values (np.ndarray): ATR values
        length (int): Period length
        multiplier (float): ATR multiplier
        
    Returns:
        tuple: (long, short, direction) arrays
        
    Performance:
        - ~5-20x faster than pure Python loops
    """
    m = len(hl2_values)
    long = np.full(m, np.nan)
    short = np.full(m, np.nan)
    direction = np.full(m, 1.0)
    
    for i in range(1, m):
        if np.isnan(atr_values[i]):
            continue
            
        hl2 = hl2_values[i]
        atr = atr_values[i]
        
        # Calculate basic bands
        _long = hl2 - multiplier * atr
        _short = hl2 + multiplier * atr
        
        # Adjust long band
        if not np.isnan(long[i - 1]):
            if _long > long[i - 1] or hl2_values[i - 1] < long[i - 1]:
                long[i] = _long
            else:
                long[i] = long[i - 1]
        else:
            long[i] = _long
        
        # Adjust short band
        if not np.isnan(short[i - 1]):
            if _short < short[i - 1] or hl2_values[i - 1] > short[i - 1]:
                short[i] = _short
            else:
                short[i] = short[i - 1]
        else:
            short[i] = _short
        
        # Determine direction
        if direction[i - 1] == 1.0:
            if hl2 <= long[i]:
                direction[i] = -1.0
            else:
                direction[i] = 1.0
        else:  # direction[i - 1] == -1.0
            if hl2 >= short[i]:
                direction[i] = 1.0
            else:
                direction[i] = -1.0
    
    return long, short, direction


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
    
    if status['available']:
        print(f"Numba Version: {status['version']}")
        if status['threading_layer']:
            print(f"Threading Layer: {status['threading_layer']}")
        if status['num_threads']:
            print(f"Number of Threads: {status['num_threads']}")
        print("\nPerformance optimizations are ENABLED")
        print("JIT-compiled functions will be used for better performance.")
    else:
        print("\nPerformance optimizations are DISABLED")
        print("Install numba for better performance: pip install numba")
        print("Functions will fall back to pure Python/NumPy implementations.")
    print("=" * 60)


# Module-level exports
__all__ = [
    "NUMBA_AVAILABLE",
    "rsx_numba_core",
    "supertrend_numba_core",
    "get_numba_status",
    "print_numba_info",
]
