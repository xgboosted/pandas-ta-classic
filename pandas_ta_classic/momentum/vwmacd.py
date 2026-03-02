# Volume Weighted Moving Average Convergence Divergence (Volume Weighted MACD)
from typing import Any, Optional
from pandas import DataFrame, Series
from pandas_ta_classic.overlap.vwma import vwma
from pandas_ta_classic.utils import (
    _swap_fast_slow,
    _build_dataframe,
    get_offset,
    verify_series,
)


def vwmacd(
    close: Series,
    volume: Series,
    fast: Optional[int] = None,
    slow: Optional[int] = None,
    signal: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: Volume Weighted MACD (VWMACD)"""
    # Validate arguments
    fast = int(fast) if fast and fast > 0 else 12
    slow = int(slow) if slow and slow > 0 else 26
    signal = int(signal) if signal and signal > 0 else 9
    fast, slow = _swap_fast_slow(fast, slow)
    close = verify_series(close, max(fast, slow, signal))
    volume = verify_series(volume, max(fast, slow, signal))
    offset = get_offset(offset)

    if close is None or volume is None:
        return None

    # Calculate Result
    # Volume weighted moving averages
    fast_vwma = vwma(close, volume, length=fast)
    slow_vwma = vwma(close, volume, length=slow)

    # VWMACD line
    vwmacd = fast_vwma - slow_vwma

    # Signal line
    signal_line = vwma(vwmacd, volume, length=signal)

    # Histogram
    histogram = vwmacd - signal_line

    # Offset + Name + Category + DataFrame
    _props = f"_{fast}_{slow}_{signal}"
    return _build_dataframe(
        {
            f"VWMACD{_props}": vwmacd,
            f"VWMACDh{_props}": histogram,
            f"VWMACDs{_props}": signal_line,
        },
        f"VWMACD{_props}",
        "momentum",
        offset,
        **kwargs,
    )


vwmacd.__doc__ = """Volume Weighted MACD (VWMACD)

Volume Weighted MACD is a variation of the traditional MACD that incorporates
volume into the calculation. It uses Volume Weighted Moving Averages (VWMA)
instead of EMAs to give more weight to periods with higher volume.

Sources:
    https://www.tradingview.com/script/NUs1Y5V7-Volume-Weighted-MACD/
    Technical Analysis Using Multiple Timeframes by Brian Shannon

Calculation:
    Default Inputs:
        fast=12, slow=26, signal=9

    FastVWMA = VWMA(close, volume, fast)
    SlowVWMA = VWMA(close, volume, slow)

    VWMACD = FastVWMA - SlowVWMA
    Signal = VWMA(VWMACD, volume, signal)
    Histogram = VWMACD - Signal

Args:
    close (pd.Series): Series of 'close's
    volume (pd.Series): Series of 'volume's
    fast (int): The fast period. Default: 12
    slow (int): The slow period. Default: 26
    signal (int): The signal period. Default: 9
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: VWMACD, Signal, and Histogram columns.
"""
