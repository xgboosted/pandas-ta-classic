# Efficiency Ratio (ER)
from typing import Any

from pandas import DataFrame, Series, concat

from pandas_ta_classic.utils import (
    apply_fill,
    apply_offset,
    degenerate_div,
    get_drift,
    get_offset,
    signals,
    verify_series,
)
from pandas_ta_classic.utils._core import _bool_param, _pos_int, nan_on_short_input


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

    # A window with no movement has no volatility to divide by, so this is
    # 0/0. degenerate_div masks it to 0.0 (TA-Lib's degenerate marker); a
    # nonzero numerator over a zero denominator (drift >= 2 over an exactly
    # periodic close) is a real x/0 and reads inf.
    denominator = abs_volatility.rolling(window=length).sum()
    er = degenerate_div(abs_diff, denominator)

    # Name it here: the signals below take their column names from it, and `.name`
    # survives the shift while a custom attribute such as `.category` does not.
    er.name = f"ER_{length}"

    # The signals read the unoffset, unfilled ER and are offset once, inside
    # signals(). Reading the shifted series shifted them a second time.
    signal_indicators = _bool_param(kwargs.pop("signal_indicators", None), False, "signal_indicators")
    signal_df = (
        signals(
            indicator=er,
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
    er = apply_offset(er, offset)

    # Categorize it
    er.category = "momentum"

    result: Series | DataFrame = er
    if signal_df is not None:
        result = concat([DataFrame({er.name: er}), signal_df], axis=1)
        # concat builds a new frame, so the name and category are set on it
        result.name = er.name
        result.category = er.category

    # The fill runs last, over every column returned, so the signal columns are
    # filled too. Until 0.9.0 it ran on the indicator alone, and an offset call
    # with fillna left NaN behind in the signal columns.
    apply_fill(result, **kwargs)

    return result


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
