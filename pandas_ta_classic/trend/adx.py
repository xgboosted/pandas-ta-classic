# -*- coding: utf-8 -*-
# Average Directional Movement Index (ADX)
from typing import Any, Optional
import numpy as np
from pandas import DataFrame, Series
from pandas_ta_classic import Imports
from pandas_ta_classic.overlap.ma import ma
from pandas_ta_classic.volatility import atr
from pandas_ta_classic.utils import (
    apply_fill,
    apply_offset,
    get_drift,
    get_offset,
    verify_series,
    zero,
)


def adx(
    high: Series,
    low: Series,
    close: Series,
    length: Optional[int] = None,
    lensig: Optional[int] = None,
    scalar: Optional[float] = None,
    mamode: Optional[str] = None,
    talib: Optional[bool] = None,
    drift: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: ADX"""
    # Validate Arguments
    length = length if length and length > 0 else 14
    lensig = lensig if lensig and lensig > 0 else length
    mamode = mamode if isinstance(mamode, str) else "rma"
    scalar = float(scalar) if scalar else 100
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    drift = get_drift(drift)
    offset = get_offset(offset)
    mode_talib = bool(talib) if isinstance(talib, bool) else False

    if high is None or low is None or close is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_talib:
        from talib import ADX, MINUS_DI, PLUS_DI

        adx_arr = Series(ADX(high, low, close, length), index=close.index)
        dmp = Series(PLUS_DI(high, low, close, length), index=close.index)
        dmn = Series(MINUS_DI(high, low, close, length), index=close.index)
    else:
        if mamode.lower() == "rma":
            # TA-Lib-compatible two-tier Wilder's smoothing for ADX.
            #
            # TA-Lib initialization:
            #   Seed Wilder sums with bars 1..length-1, then apply the first
            #   Wilder update at bar `length`. This matches TA-Lib's internal
            #   lookback behavior and produces an exact numerical match.
            #
            # DX computation: DX = 100 * |dm_pos - dm_neg| / (dm_pos + dm_neg)
            #   where dm_pos/dm_neg/tr are Wilder-smoothed running sums.
            #
            # ADX computation (two-tier):
            #   1. Seed ADX = SMA of the first `lensig` DX values (at bar `length`)
            #   2. Wilder's RMA from bar `length+lensig` onward:
            #      adx = adx_prev + (1/lensig) * (dx - adx_prev)
            h_arr = high.values.astype(float)
            l_arr = low.values.astype(float)
            c_arr = close.values.astype(float)
            n = len(h_arr)

            dm_pos = np.zeros(n)
            dm_neg = np.zeros(n)
            tr_raw = np.zeros(n)
            for idx in range(1, n):
                up = h_arr[idx] - h_arr[idx - 1]
                dn = l_arr[idx - 1] - l_arr[idx]
                dm_pos[idx] = up if (up > dn and up > 0) else 0.0
                dm_neg[idx] = dn if (dn > up and dn > 0) else 0.0
                tr_raw[idx] = max(
                    h_arr[idx] - l_arr[idx],
                    abs(h_arr[idx] - c_arr[idx - 1]),
                    abs(l_arr[idx] - c_arr[idx - 1]),
                )

            # Seed: sum of bars 1..length-1, then first Wilder update at bar `length`
            tr14 = tr_raw[1:length].sum()
            dmpos14 = dm_pos[1:length].sum()
            dmneg14 = dm_neg[1:length].sum()
            tr14 = tr14 - tr14 / length + tr_raw[length]
            dmpos14 = dmpos14 - dmpos14 / length + dm_pos[length]
            dmneg14 = dmneg14 - dmneg14 / length + dm_neg[length]

            if n <= length:
                return None

            # Compute DX, DMP, DMN for all bars from `length` onward.
            # Using the same Wilder running sums for DMP/DMN ensures identical
            # seeding as the ADX column (avoids the different-init issue that
            # arises when ma("rma", ...) is called separately).
            dx_raw = np.full(n, np.nan)
            dmp_raw = np.full(n, np.nan)
            dmn_raw = np.full(n, np.nan)
            denom = dmpos14 + dmneg14
            dx_raw[length] = (
                scalar * abs(dmpos14 - dmneg14) / denom if denom != 0 else 0.0
            )
            dmp_raw[length] = scalar * dmpos14 / tr14 if tr14 != 0 else 0.0
            dmn_raw[length] = scalar * dmneg14 / tr14 if tr14 != 0 else 0.0

            _tr14, _dmpos14, _dmneg14 = tr14, dmpos14, dmneg14
            for idx in range(length + 1, n):
                _tr14 = _tr14 - _tr14 / length + tr_raw[idx]
                _dmpos14 = _dmpos14 - _dmpos14 / length + dm_pos[idx]
                _dmneg14 = _dmneg14 - _dmneg14 / length + dm_neg[idx]
                denom = _dmpos14 + _dmneg14
                dx_raw[idx] = (
                    scalar * abs(_dmpos14 - _dmneg14) / denom if denom != 0 else 0.0
                )
                dmp_raw[idx] = scalar * _dmpos14 / _tr14 if _tr14 != 0 else 0.0
                dmn_raw[idx] = scalar * _dmneg14 / _tr14 if _tr14 != 0 else 0.0

            # Two-tier ADX: seed = SMA of first `lensig` DX values, then Wilder's RMA
            dx_valid = dx_raw[length:]  # first valid DX is at index `length`
            if len(dx_valid) < lensig:
                return None

            adx_seed = dx_valid[:lensig].mean()
            alpha = 1.0 / lensig
            adx_values = [adx_seed]
            for idx in range(lensig, len(dx_valid)):
                adx_values.append(
                    adx_values[-1] + alpha * (dx_valid[idx] - adx_values[-1])
                )

            # ADX first valid at index `length + lensig - 1` in the original array
            adx_start = length + lensig - 1
            adx_raw = np.full(n, np.nan)
            adx_raw[adx_start : adx_start + len(adx_values)] = adx_values
            adx_arr = Series(adx_raw, index=close.index, dtype=float)

            dmp = Series(dmp_raw, index=close.index, dtype=float)
            dmn = Series(dmn_raw, index=close.index, dtype=float)
        else:
            atr_ = atr(high=high, low=low, close=close, length=length, talib=False)
            if atr_ is None:
                return None

            up = high - high.shift(drift)  # high.diff(drift)
            dn = low.shift(drift) - low  # low.diff(-drift).shift(drift)

            pos = ((up > dn) & (up > 0)) * up
            neg = ((dn > up) & (dn > 0)) * dn

            pos = pos.apply(zero)
            neg = neg.apply(zero)

            k = scalar / atr_
            _dmp_ma = ma(mamode, pos, length=length)
            if _dmp_ma is None:
                return None
            _dmn_ma = ma(mamode, neg, length=length)
            if _dmn_ma is None:
                return None
            dmp = k * _dmp_ma
            dmn = k * _dmn_ma

            dx = scalar * (dmp - dmn).abs() / (dmp + dmn)
            adx_arr = ma(mamode, dx, length=lensig)
            if adx_arr is None:
                return None

    # Offset
    dmp, dmn, adx_arr = apply_offset([dmp, dmn, adx_arr], offset)

    adx_arr, dmp, dmn = apply_fill([adx_arr, dmp, dmn], **kwargs)

    # Name and Categorize it
    adx_arr.name = f"ADX_{lensig}"
    dmp.name = f"DMP_{length}"
    dmn.name = f"DMN_{length}"

    adx_arr.category = dmp.category = dmn.category = "trend"

    # Prepare DataFrame to return
    data = {adx_arr.name: adx_arr, dmp.name: dmp, dmn.name: dmn}
    adxdf = DataFrame(data)
    adxdf.name = f"ADX_{lensig}"
    adxdf.category = "trend"

    return adxdf


adx.__doc__ = """Average Directional Movement (ADX)

Average Directional Movement is meant to quantify trend strength by measuring
the amount of movement in a single direction.

Sources:
    https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/average-directional-movement-adx/
    TA Lib Correlation: ~100% (exact match when mamode='rma')

Calculation:
    DMI ADX TREND 2.0 by @TraderR0BERT, NETWORTHIE.COM
        //Created by @TraderR0BERT, NETWORTHIE.COM, last updated 01/26/2016
        //DMI Indicator
        //Resolution input option for higher/lower time frames
        study(title="DMI ADX TREND 2.0", shorttitle="ADX TREND 2.0")

        adxlen = input(14, title="ADX Smoothing")
        dilen = input(14, title="DI Length")
        thold = input(20, title="Threshold")

        threshold = thold

        //Script for Indicator
        dirmov(len) =>
            up = change(high)
            down = -change(low)
            truerange = rma(tr, len)
            plus = fixnan(100 * rma(up > down and up > 0 ? up : 0, len) / truerange)
            minus = fixnan(100 * rma(down > up and down > 0 ? down : 0, len) / truerange)
            [plus, minus]

        adx(dilen, adxlen) =>
            [plus, minus] = dirmov(dilen)
            sum = plus + minus
            adx = 100 * rma(abs(plus - minus) / (sum == 0 ? 1 : sum), adxlen)
            [adx, plus, minus]

        [sig, up, down] = adx(dilen, adxlen)
        osob=input(40,title="Exhaustion Level for ADX, default = 40")
        col = sig >= sig[1] ? green : sig <= sig[1] ? red : gray

        //Plot Definitions Current Timeframe
        p1 = plot(sig, color=col, linewidth = 3, title="ADX")
        p2 = plot(sig, color=col, style=circles, linewidth=3, title="ADX")
        p3 = plot(up, color=blue, linewidth = 3, title="+DI")
        p4 = plot(up, color=blue, style=circles, linewidth=3, title="+DI")
        p5 = plot(down, color=fuchsia, linewidth = 3, title="-DI")
        p6 = plot(down, color=fuchsia, style=circles, linewidth=3, title="-DI")
        h1 = plot(threshold, color=black, linewidth =3, title="Threshold")

        trender = (sig >= up or sig >= down) ? 1 : 0
        bgcolor(trender>0?black:gray, transp=85)

        //Alert Function for ADX crossing Threshold
        Up_Cross = crossover(up, threshold)
        alertcondition(Up_Cross, title="DMI+ cross", message="DMI+ Crossing Threshold")
        Down_Cross = crossover(down, threshold)
        alertcondition(Down_Cross, title="DMI- cross", message="DMI- Crossing Threshold")

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 14
    lensig (int): Signal Length. Like TradingView's default ADX. Default: length
    scalar (float): How much to magnify. Default: 100
    mamode (str): See ```help(ta.ma)```. Default: 'rma'
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    drift (int): The difference period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: adx, dmp, dmn columns.
"""
