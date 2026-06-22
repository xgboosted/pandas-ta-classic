import pandas as pd
import numpy as np

def smc_sweep(open_, high, low, close, length=15, wick_mult=1.5, **kwargs):
    """Indicator: Smart Money Concept Liquidity Sweep"""
    # 1. Validate Arguments
    length = int(length) if length and length > 0 else 15
    wick_mult = float(wick_mult) if wick_mult and wick_mult > 0 else 1.5

    # 2. Structural Baselines
    swing_low = low.rolling(window=length).min().shift(1)
    swing_high = high.rolling(window=length).max().shift(1)
    
    body = (close - open_).abs()
    
    # Vectorized Wicks
    lower_wick = pd.concat([open_, close], axis=1).min(axis=1) - low
    upper_wick = high - pd.concat([open_, close], axis=1).max(axis=1)

    # 3. Bullish Sweep Logic (+1)
    bull_sweep = np.where(
        (low < swing_low) & 
        (close > swing_low) & 
        (close > open_) &
        (lower_wick > (body * wick_mult)), 
        1, 0
    )

    # 4. Bearish Sweep Logic (-1)
    bear_sweep = np.where(
        (high > swing_high) & 
        (close < swing_high) & 
        (close < open_) &
        (upper_wick > (body * wick_mult)), 
        -1, 0
    )

    # Combine signals: 1 for Bull, -1 for Bear, 0 for None
    combined_signal = bull_sweep + bear_sweep

    # 5. Prepare DataFrame to return
    _props = f"_{length}_{wick_mult}"
    sweep_series = pd.Series(combined_signal, index=close.index, name=f"SMC_SWEEP{_props}")
    
    return sweep_series

smc_sweep.__doc__ = \
"""Smart Money Concept Liquidity Sweep

Identifies when price drops below a swing low (or above a swing high), 
violently rejects it, leaving a long wick and closing in the opposite direction.
Returns 1 for a Bullish Sweep, -1 for a Bearish Sweep, and 0 for no signal.

Calculation:
    Default Inputs:
        length=15, wick_mult=1.5
    
Args:
    open_ (pd.Series): Series of 'open's
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): Swing high/low lookback period. Default: 15
    wick_mult (float): Multiplier for wick vs body ratio. Default: 1.5

Returns:
    pd.Series: 1 (Bullish), -1 (Bearish), or 0.
"""