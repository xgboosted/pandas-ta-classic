# Stochastic Momentum Index (SMI)
from typing import Any, Optional
from pandas import DataFrame, Series
from .tsi import tsi
from pandas_ta_classic.utils import (
    _swap_fast_slow,
    _build_dataframe,
    get_offset,
    verify_series,
)


def smi(
    close: Series,
    fast: Optional[int] = None,
    slow: Optional[int] = None,
    signal: Optional[int] = None,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: SMI Ergodic Indicator (SMIIO)"""
    # Validate arguments
    fast = int(fast) if fast and fast > 0 else 5
    slow = int(slow) if slow and slow > 0 else 20
    signal = int(signal) if signal and signal > 0 else 5
    fast, slow = _swap_fast_slow(fast, slow)
    scalar = float(scalar) if scalar else 1
    close = verify_series(close, max(fast, slow, signal))
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result
    tsi_df = tsi(close, fast=fast, slow=slow, signal=signal, scalar=scalar)
    smi = tsi_df.iloc[:, 0]
    signalma = tsi_df.iloc[:, 1]
    osc = smi - signalma

    # Offset + Name + Category + DataFrame
    _scalar = f"_{scalar}" if scalar != 1 else ""
    _props = f"_{fast}_{slow}_{signal}{_scalar}"
    return _build_dataframe(
        {f"SMI{_props}": smi, f"SMIs{_props}": signalma, f"SMIo{_props}": osc},
        f"SMI{_props}",
        "momentum",
        offset,
        **kwargs,
    )


smi.__doc__ = """SMI Ergodic Indicator (SMI)

The SMI Ergodic Indicator is the same as the True Strength Index (TSI) developed
by William Blau, except the SMI includes a signal line. The SMI uses double
moving averages of price minus previous price over 2 time frames. The signal
line, which is an EMA of the SMI, is plotted to help trigger trading signals.
The trend is bullish when crossing above zero and bearish when crossing below
zero. This implementation includes both the SMI Ergodic Indicator and SMI
Ergodic Oscillator.

Sources:
    https://www.motivewave.com/studies/smi_ergodic_indicator.htm
    https://www.tradingview.com/script/Xh5Q0une-SMI-Ergodic-Oscillator/
    https://www.tradingview.com/script/cwrgy4fw-SMIIO/

Calculation:
    Default Inputs:
        fast=5, slow=20, signal=5
    TSI = True Strength Index
    EMA = Exponential Moving Average

    ERG = TSI(close, fast, slow)
    Signal = EMA(ERG, signal)
    OSC = ERG - Signal

Args:
    close (pd.Series): Series of 'close's
    fast (int): The short period. Default: 5
    slow (int): The long period. Default: 20
    signal (int): The signal period. Default: 5
    scalar (float): How much to magnify. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: smi, signal, oscillator columns.
"""
