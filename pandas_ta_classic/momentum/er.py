# Efficiency Ratio (ER)
from typing import Any

from pandas import DataFrame, Series

from pandas_ta_classic.utils import get_drift, get_offset, verify_series
from pandas_ta_classic.utils._core import _pos_int, nan_on_short_input
from pandas_ta_classic.utils._signals import attach_signals


@nan_on_short_input
def er(
    close: Series,
    length: int | None = None,
    drift: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | DataFrame | None:
    """Indicator: Efficiency Ratio (ER)"""
    # Validate arguments
    length = _pos_int(length, 10, "length")
    close = verify_series(close, length)
    offset = get_offset(offset)
    drift = get_drift(drift)

    if close is None:
        return None

    # Calculate Result
    abs_diff = close.diff(length).abs()
    abs_volatility = close.diff(drift).abs()

    er = abs_diff
    er /= abs_volatility.rolling(window=length).sum()

    # Name it here: the signal columns take their names from it, and `.name`
    # survives the shift while a custom attribute such as `.category` does not.
    er.name = f"ER_{length}"

    # attach_signals owns the order of the remaining steps (signals off the
    # unoffset series, one shift, then the fill over every column).
    return attach_signals(er, category="momentum", offset=offset, kwargs=kwargs)


er.__doc__ = """Efficiency Ratio (ER)

The Efficiency Ratio was invented by Perry J. Kaufman and presented in his book "New Trading Systems and Methods". It is designed to account for market noise or volatility.

It is calculated by dividing the net change in price movement over N periods by the sum of the absolute net changes over the same N periods.

Sources:
    https://help.tc2000.com/m/69404/l/749623-kaufman-efficiency-ratio

Calculation:
    Default Inputs:
        length=10
    ABS = Absolute Value
    EMA = Exponential Moving Average

    abs_diff = ABS(close.diff(length))
    volatility = ABS(close.diff(1))
    ER = abs_diff / SUM(volatility, length)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 10
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
