# Relative Strength Index (RSI)
from typing import Any

from pandas import DataFrame, Series

from pandas_ta_classic import Imports
from pandas_ta_classic.overlap.rma import rma
from pandas_ta_classic.utils import get_drift, get_offset, verify_series
from pandas_ta_classic.utils._core import _bool_param, _number, _pos_int, nan_on_short_input
from pandas_ta_classic.utils._signals import attach_signals


@nan_on_short_input
def rsi(
    close: Series,
    length: int | None = None,
    scalar: float | None = None,
    talib: bool | None = None,
    drift: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | DataFrame | None:
    """Indicator: Relative Strength Index (RSI)"""
    # Validate arguments
    length = _pos_int(length, 14, "length")
    scalar = _number(scalar, 100, "scalar")
    close = verify_series(close, length)
    drift = get_drift(drift)
    offset = get_offset(offset)
    mode_talib = _bool_param(talib, False, "talib")

    if close is None:
        return None

    # Calculate Result
    # TA-Lib cannot express a non-default drift, scalar; run natively instead of ignoring it
    if Imports["talib"] and mode_talib and scalar == 100 and drift == 1:
        from talib import RSI

        rsi = RSI(close, length)
    else:
        negative = close.diff(drift)
        positive = negative.copy()

        positive[positive < 0] = 0  # Make negatives 0 for the postive series
        negative[negative > 0] = 0  # Make postives 0 for the negative series

        positive_avg = rma(positive, length=length)
        negative_avg = rma(negative, length=length)

        rsi = scalar * positive_avg / (positive_avg + negative_avg.abs())

    # Name it here: the signal columns take their names from it, and `.name`
    # survives the shift while a custom attribute such as `.category` does not.
    rsi.name = f"RSI_{length}"

    # attach_signals owns the order of the remaining steps (signals off the
    # unoffset series, one shift, then the fill over every column).
    return attach_signals(rsi, category="momentum", offset=offset, kwargs=kwargs)


rsi.__doc__ = """Relative Strength Index (RSI)

The Relative Strength Index is popular momentum oscillator used to measure the
velocity as well as the magnitude of directional price movements.

Sources:
    https://www.tradingview.com/wiki/Relative_Strength_Index_(RSI)

Calculation:
    Default Inputs:
        length=14, scalar=100, drift=1
    ABS = Absolute Value
    RMA = Rolling Moving Average

    diff = close.diff(drift)
    positive = diff if diff > 0 else 0
    negative = diff if diff < 0 else 0

    pos_avg = RMA(positive, length)
    neg_avg = ABS(RMA(negative, length))

    RSI = scalar * pos_avg / (pos_avg + neg_avg)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 14
    scalar (float): How much to magnify. Default: 100
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: False
    drift (int): The difference period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    signal_indicators (bool): When True, threshold and comparison signal
        columns are appended and the result becomes a DataFrame instead of a
        Series. The options below are only read when it is True. Default: False
    xa (float): Upper threshold. Default: 80
    xb (float): Lower threshold. Default: 20
    cross_values (bool): When True, the xa/xb columns mark the bars that cross
        the threshold instead of flagging every bar on one side of it.
        Default: False
    xserie (pd.Series): Comparison series; used for both xserie_a and xserie_b
        unless one of them is given. Default: None
    xserie_a (pd.Series): Comparison series for the "above" column.
        Default: None
    xserie_b (pd.Series): Comparison series for the "below" column.
        Default: None
    cross_series (bool): When True, the xserie columns mark crossings instead
        of flagging every bar on one side of the series. Default: True
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated, or a pd.DataFrame with the indicator and
        its signal columns when signal_indicators is True.
"""
