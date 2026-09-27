# Relative Strength Index (RSI)
from typing import Any

from pandas import DataFrame, Series, concat

from pandas_ta_classic import Imports
from pandas_ta_classic.overlap.rma import rma
from pandas_ta_classic.utils import (
    apply_fill,
    apply_offset,
    get_drift,
    get_offset,
    signals,
    verify_series,
)
from pandas_ta_classic.utils._core import _bool_param, _number, _pos_int, nan_on_short_input


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

    # Name it here: the signals below take their column names from it, and `.name`
    # survives the shift while a custom attribute such as `.category` does not.
    rsi.name = f"RSI_{length}"

    # The signals read the unoffset, unfilled RSI and are offset once, inside
    # signals(). Reading the shifted series shifted them a second time.
    signal_indicators = _bool_param(kwargs.pop("signal_indicators", None), False, "signal_indicators")
    signal_df = (
        signals(
            indicator=rsi,
            xa=kwargs.pop("xa", 80),
            xb=kwargs.pop("xb", 20),
            xserie=kwargs.pop("xserie", None),
            xserie_a=kwargs.pop("xserie_a", None),
            xserie_b=kwargs.pop("xserie_b", None),
            cross_values=_bool_param(kwargs.pop("cross_values", None), False, "cross_values"),
            cross_series=_bool_param(kwargs.pop("cross_series", None), True, "cross_series"),
            offset=offset,
        )
        if signal_indicators
        else None
    )

    # Offset
    rsi = apply_offset(rsi, offset)

    # Categorize it
    rsi.category = "momentum"

    result: Series | DataFrame = rsi
    if signal_df is not None:
        result = concat([DataFrame({rsi.name: rsi}), signal_df], axis=1)
        # concat builds a new frame, so the name and category are set on it
        result.name = rsi.name
        result.category = rsi.category

    # The fill runs last, over every column returned, so the signal columns are
    # filled too. Until 0.9.0 it ran on the indicator alone, and an offset call
    # with fillna left NaN behind in the signal columns.
    apply_fill(result, **kwargs)

    return result


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
