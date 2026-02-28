# -*- coding: utf-8 -*-
# TRIX Histogram (TRIXH)
from typing import Any, Optional
from pandas import DataFrame, Series
from pandas_ta_classic.momentum import trix
from pandas_ta_classic.utils import apply_offset, get_offset, verify_series


def trixh(
    close: Series,
    length: Optional[int] = None,
    signal: Optional[int] = None,
    scalar: Optional[float] = None,
    drift: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: TRIX Histogram (TRIXH)"""
    # Validate arguments
    length = int(length) if length and length > 0 else 18
    signal = int(signal) if signal and signal > 0 else 9
    scalar = float(scalar) if scalar else 100
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result
    # Calculate TRIX (returns DataFrame with TRIX and signal)
    trix_df = trix(close, length=length, signal=signal, scalar=scalar, drift=drift)

    if trix_df is None:
        return None

    # Extract TRIX line and signal
    trix_col = f"TRIX_{length}_{signal}"
    signal_col = f"TRIXs_{length}_{signal}"

    trix_line = trix_df[trix_col]
    trix_signal = trix_df[signal_col]

    # Calculate histogram
    histogram = trix_line - trix_signal

    # Offset
    trix_line = apply_offset(trix_line, offset, **kwargs)
    trix_signal = apply_offset(trix_signal, offset, **kwargs)
    histogram = apply_offset(histogram, offset, **kwargs)

    # Name and Categorize it
    trix_line.name = f"TRIX_{length}_{signal}"
    trix_signal.name = f"TRIXs_{length}_{signal}"
    histogram.name = f"TRIXh_{length}_{signal}"
    trix_line.category = trix_signal.category = histogram.category = "momentum"

    # Prepare DataFrame to return
    data = {
        trix_line.name: trix_line,
        trix_signal.name: trix_signal,
        histogram.name: histogram,
    }
    df = DataFrame(data)
    df.name = f"TRIXH_{length}_{signal}"
    df.category = "momentum"

    return df


trixh.__doc__ = """TRIX Histogram (TRIXH)

TRIX Histogram extends the TRIX indicator by adding a signal line and histogram.
The histogram represents the difference between TRIX and its signal line,
similar to MACD histogram, helping identify momentum changes and divergences.

Sources:
    https://www.investopedia.com/terms/t/trix.asp
    https://school.stockcharts.com/doku.php?id=technical_indicators:trix

Calculation:
    Default Inputs:
        length=18, signal=9, scalar=100

    TRIX = TRIX(close, length, scalar)
    Signal = EMA(TRIX, signal)
    Histogram = TRIX - Signal

Args:
    close (pd.Series): Series of 'close's
    length (int): TRIX period. Default: 18
    signal (int): Signal line period. Default: 9
    scalar (float): Multiplier. Default: 100
    drift (int): The difference period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: TRIX, Signal, and Histogram columns.
"""
