# Volume Flow Indicator (VFI)
from typing import Any

import numpy as np
from pandas import Series

from pandas_ta_classic.overlap.hlc3 import hlc3
from pandas_ta_classic.overlap.ma import ma
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _number, _pos_int, _str_param, degenerate_div, nan_on_short_input

# The source fixes the volatility window at 30 bars, independent of `length`.
_VINTER_LENGTH = 30


@nan_on_short_input
def vfi(
    high: Series,
    low: Series,
    close: Series,
    volume: Series,
    length: int | None = None,
    coef: float | None = None,
    vcoef: float | None = None,
    mamode: str | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Volume Flow Indicator (VFI)"""
    # Validate arguments
    length = _pos_int(length, 130, "length")
    coef = _number(coef, 0.2, "coef")
    vcoef = _number(vcoef, 2.5, "vcoef")
    mamode = _str_param(mamode, "ema", "mamode")
    _length = max(length, _VINTER_LENGTH) + 1
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    close = verify_series(close, _length)
    volume = verify_series(volume, _length)
    offset = get_offset(offset)

    if high is None or low is None or close is None or volume is None:
        return None

    # Calculate Result, as Katsanos and LazyBear define it. The previous version
    # used the close for the typical price, a cutoff of coef * close (a 20 % move
    # at the default coef, so VFI was 0.0 on every SPY_D bar), volume times the
    # price change instead of signed volume, and divided by an average of vave.
    typical = hlc3(high, low, close)
    inter = np.log(typical).diff().to_numpy(dtype=float)
    # Population deviation (Pine's stdev) of the log change. Pure numpy: an
    # all-finite window is required, so a missing bar leaves its windows NaN.
    vinter = np.full(inter.size, np.nan)
    if _VINTER_LENGTH <= inter.size:
        vinter[_VINTER_LENGTH - 1 :] = np.lib.stride_tricks.sliding_window_view(inter, _VINTER_LENGTH).std(axis=1)
    cutoff = coef * vinter * close.to_numpy(dtype=float)

    vave = volume.rolling(length).mean().shift(1)
    # np.minimum, not clip(upper=): clip ignores a NaN bound and passed the
    # uncapped volume through wherever vave was missing.
    vc = np.minimum(volume.to_numpy(dtype=float), vave.to_numpy(dtype=float) * vcoef)
    mf = typical.diff().to_numpy(dtype=float)

    # Signed, capped volume where the typical price moved past the cutoff, 0
    # inside it. NaN where any operand is missing: a comparison with NaN is
    # False and would read as "no flow".
    vcp = np.where(mf > cutoff, vc, np.where(mf < -cutoff, -vc, 0.0))
    vcp[np.isnan(mf) | np.isnan(cutoff) | np.isnan(vc)] = np.nan
    vcp = Series(vcp, index=close.index)

    # A flat series has no flow and, with no volume, no average: 0/0 reads 0.0
    # (tests/test_degenerate_input.py); any other zero average is a real
    # division by zero.
    vfi = degenerate_div(vcp.rolling(length).sum(), vave)

    # Smooth VFI
    vfi = ma(mamode, vfi, length=3)
    if vfi is None:
        return None

    # Offset
    vfi = apply_offset(vfi, offset)

    vfi = apply_fill(vfi, **kwargs)

    # Name and Categorize it
    vfi.name = f"VFI_{length}"
    vfi.category = "volume"

    return vfi


vfi.__doc__ = """Volume Flow Indicator (VFI)

The Volume Flow Indicator (VFI) is a volume-based indicator that helps identify
the strength of bulls vs bears in the market. It combines price movement with
volume to show the flow of money into or out of a security.

Sources:
    Markos Katsanos, "Volume Flow Indicator", Technical Analysis of Stocks & Commodities, June 2004
    https://www.tradingview.com/script/MhlDpfdS-Volume-Flow-Indicator-LazyBear/
    https://www.investopedia.com/terms/v/volume-analysis.asp

Calculation:
    Default Inputs:
        length=130, coef=0.2, vcoef=2.5, mamode='ema'

    typical = HLC3
    inter = LOG(typical) - LOG(typical.shift(1))
    vinter = STDEV(inter, 30)  # population deviation, fixed 30 bars
    cutoff = coef * vinter * close

    vave = SMA(volume, length).shift(1)
    vmax = vave * vcoef
    vc = MIN(volume, vmax)

    mf = typical - typical.shift(1)
    vcp = vc if mf > cutoff else -vc if mf < -cutoff else 0

    VFI = SUM(vcp, length) / vave
    VFI = MA(mamode, VFI, 3)  # Katsanos' 3-bar EMA; LazyBear's script leaves it off by default

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    volume (pd.Series): Series of 'volume's
    length (int): The period. Default: 130
    coef (float): Volatility threshold coefficient (0.2 for day trading, 0.1 for intra-day). Default: 0.2
    vcoef (float): Volume coefficient. Default: 2.5
    mamode (str): Moving average mode for smoothing. Default: 'ema'
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
