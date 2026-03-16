# -*- coding: utf-8 -*-
# Volume Flow Indicator (VFI)
from typing import Any, Optional
from pandas import Series
from pandas_ta_classic.overlap.ma import ma
from pandas_ta_classic.utils import get_offset, verify_series


def vfi(
    close: Series,
    volume: Series,
    length: Optional[int] = None,
    coef: Optional[float] = None,
    vcoef: Optional[float] = None,
    mamode: Optional[str] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Volume Flow Indicator (VFI)"""
    # Validate arguments
    length = int(length) if length and length > 0 else 130
    coef = float(coef) if coef else 0.2
    vcoef = float(vcoef) if vcoef else 2.5
    mamode = mamode.lower() if mamode and isinstance(mamode, str) else "ema"
    _length = length
    close = verify_series(close, _length)
    volume = verify_series(volume, _length)
    offset = get_offset(offset)

    if close is None or volume is None:
        return None

    # Calculate Result
    # Typical price
    typical = close

    # Volume cutoff
    vave = volume.rolling(length).mean().shift(1)
    vmax = vave * vcoef
    vc = volume.clip(upper=vmax)

    # Calculate MF (Money Flow) with volatility threshold
    # Only consider price changes above the threshold
    inter = typical - typical.shift(1)

    # Apply volatility threshold: coef * close
    cutoff = coef * close
    mf = inter.where(inter.abs() > cutoff, 0)

    # VCP (Volume times Cutoff Price)
    vcp = vc * mf

    # Calculate VFI (protect against division by zero)
    vave_mean = vave.rolling(length).mean()
    vfi = vcp.rolling(length).sum() / vave_mean.replace(0, float("nan"))

    # Smooth VFI
    vfi = ma(mamode, vfi, length=3)
    if vfi is None:
        return None

    # Offset
    if offset != 0:
        vfi = vfi.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        vfi.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        if kwargs["fill_method"] == "ffill":
            vfi.ffill(inplace=True)
        elif kwargs["fill_method"] == "bfill":
            vfi.bfill(inplace=True)

    # Name and Categorize it
    vfi.name = f"VFI_{length}"
    vfi.category = "volume"

    return vfi


vfi.__doc__ = """Volume Flow Indicator (VFI)

The Volume Flow Indicator (VFI) is a volume-based indicator that helps identify
the strength of bulls vs bears in the market. It combines price movement with
volume to show the flow of money into or out of a security.

Sources:
    https://www.tradingview.com/script/MhlDpfdS-Volume-Flow-Indicator-LazyBear/
    https://www.investopedia.com/terms/v/volume-analysis.asp

Calculation:
    Default Inputs:
        length=130, coef=0.2, vcoef=2.5, mamode='ema'

    typical = close
    inter = typical - typical.shift(1)  # Price change
    cutoff = coef * close  # Volatility threshold
    mf = inter if abs(inter) > cutoff else 0  # Filter minimal price changes
    
    vave = SMA(volume, length).shift(1)
    vmax = vave * vcoef
    vc = min(volume, vmax)  # Clipped volume

    vcp = vc * mf  # Volume-weighted money flow

    VFI = SUM(vcp, length) / SMA(vave, length)  # Protected against division by zero
    VFI = EMA(VFI, 3)  # Smooth the result

Args:
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
