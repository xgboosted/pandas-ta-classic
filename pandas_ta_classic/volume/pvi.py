# Positive Volume Index (PVI)
from typing import Any

from pandas import Series

from pandas_ta_classic.momentum.roc import roc
from pandas_ta_classic.utils import (
    apply_fill,
    apply_offset,
    get_offset,
    signed_series,
    verify_series,
)
from pandas_ta_classic.utils._core import _name_tokens, _pos_int, _positive_inputs, nan_on_short_input, skip_leading_nan


@nan_on_short_input
@_positive_inputs("close")
@skip_leading_nan("close", "volume")
def pvi(
    close: Series,
    volume: Series,
    length: int | None = None,
    initial: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Positive Volume Index (PVI)"""
    # Validate arguments
    length = _pos_int(length, 1, "length")
    initial = _pos_int(initial, 1000, "initial")
    close = verify_series(close, length)
    volume = verify_series(volume, length)
    offset = get_offset(offset)

    if close is None or volume is None:
        return None

    # Calculate Result
    signed_volume = signed_series(volume, 1)
    # where(), not signed_volume[signed_volume > 0]: see the note in nvi.py --
    # realigning the subset rebuilds the index and clears DatetimeIndex.freq on
    # the caller's own index object.
    # Each rising-volume bar multiplies the index by 1 + its rate of change, as
    # in TA-Lib's PVI and Fosback's definition; adding the percent change made
    # it another series (#344).
    factor = 1 + roc(close=close, length=length) * signed_volume.where(signed_volume > 0).abs() / 100
    factor = factor.fillna(1.0)
    factor.iloc[0] = 1.0
    pvi = initial * factor.cumprod()

    # A bar whose input is missing has no value; the state carries across it.
    pvi = pvi.mask((close.isna() | volume.isna()).to_numpy())

    # Offset
    pvi = apply_offset(pvi, offset)

    pvi = apply_fill(pvi, **kwargs)

    # Name and Categorize it
    pvi.name = f"PVI_{length}" + _name_tokens(("initial", initial, 1000))
    pvi.category = "volume"

    return pvi


pvi.__doc__ = """Positive Volume Index (PVI)

The Positive Volume Index is a cumulative indicator that uses volume change in
an attempt to identify where smart money is active.
Used in conjunction with NVI.

Sources:
    https://www.investopedia.com/terms/p/pvi.asp

Calculation:
    Default Inputs:
        length=1, initial=1000
    ROC = Rate of Change

    roc = ROC(close, length)
    signed_volume = signed_series(volume, initial=1)
    factor = 1 + signed_volume.where(signed_volume > 0).abs() * roc / 100
    factor = factor.fillna(1); factor.iloc[0] = 1
    pvi = initial * factor.cumprod()

    Multiplicative, as in Fosback's definition and TA-Lib's PVI.

Args:
    close (pd.Series): Series of 'close's
    volume (pd.Series): Series of 'volume's
    length (int): The short period. Default: 1
    initial (int): The short period. Default: 1000
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
