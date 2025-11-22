# -*- coding: utf-8 -*-
# Volume Flow Indicator (VFI)
import numpy as np
from pandas_ta_classic.overlap.ma import ma
from pandas_ta_classic.utils import get_offset, verify_series


def vfi(
    high,
    low,
    close,
    volume,
    length=None,
    coef=None,
    vcoef=None,
    mamode=None,
    offset=None,
    **kwargs,
):
    """Indicator: Volume Flow Indicator (VFI)"""
    # Validate arguments
    length = int(length) if length and length > 0 else 130
    coef = float(coef) if coef else 0.2
    vcoef = float(vcoef) if vcoef else 2.5
    mamode = mamode.lower() if mamode and isinstance(mamode, str) else "ema"
    _length = length
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    close = verify_series(close, _length)
    volume = verify_series(volume, _length)
    offset = get_offset(offset)

    if high is None or low is None or close is None or volume is None:
        return

    # Calculate Result
    # Typical price (HLC/3)
    typical = (high + low + close) / 3.0

    # Calculate logarithmic price change
    # Replace zero or negative values with NaN to avoid log domain errors
    typical = typical.where(typical > 0, np.nan)
    log_typical = np.log(typical)
    inter = log_typical.diff()

    # Calculate standard deviation of price changes (volatility)
    vinter = inter.rolling(length).std()

    # Apply coef filter: only use price changes above threshold
    # When |inter| < coef * vinter, set to 0 (filter out noise)
    cutoff = coef * vinter
    inter = inter.where(inter.abs() >= cutoff, 0)

    # Volume cutoff
    vave = volume.rolling(length).mean()
    vmax = vave * vcoef
    vc = volume.clip(upper=vmax)

    # Calculate VCP (Volume times Cutoff Price change)
    vcp = vc * inter

    # Calculate VFI
    vfi_raw = vcp.rolling(length).sum() / vave
    
    # Smooth VFI
    vfi = ma(mamode, vfi_raw, length=3)

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
volume to show the flow of money into or out of a security. The indicator uses
a coefficient to filter out insignificant price changes based on volatility.

Sources:
    https://www.tradingview.com/script/MhlDpfdS-Volume-Flow-Indicator-LazyBear/
    http://mkatsanos.com/VFI.html
    https://www.investopedia.com/terms/v/volume-analysis.asp

Calculation:
    Default Inputs:
        length=130, coef=0.2, vcoef=2.5, mamode='ema'

    typical = (high + low + close) / 3
    inter = log(typical) - log(typical[1])
    vinter = stdev(inter, length)
    cutoff = coef * vinter
    
    # Filter price changes below threshold
    if |inter| < cutoff:
        inter = 0
    
    vave = SMA(volume, length)
    vmax = vave * vcoef
    vc = min(volume, vmax)
    
    vcp = vc * inter
    
    VFI = SUM(vcp, length) / vave
    VFI = EMA(VFI, 3)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    volume (pd.Series): Series of 'volume's
    length (int): The period. Default: 130
    coef (float): Volatility filter coefficient. Filters out price changes 
        below coef * stdev. Use 0.2 for daily, 0.1 for intraday. Default: 0.2
    vcoef (float): Volume coefficient/cutoff multiplier. Default: 2.5
    mamode (str): Moving average mode for smoothing. Default: 'ema'
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
