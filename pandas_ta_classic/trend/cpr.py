# Central Pivot Range (CPR)
from typing import Any, Optional

from pandas import DataFrame, Series
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._cpr import (
    get_previous_period_ohlcv,
    calculate_cpr_width,
    calculate_price_position,
    detect_virgin_cpr,
)


def _any_none(*args):
    return any(x is None for x in args)


def _cpr_add_level(pivot_result, data, level):
    """Add one named pivot level series to *data* if present in *pivot_result*."""
    key = level.lower()
    if key in pivot_result:
        s = pivot_result[key]
        s.name = f"CPR_{level}"
        s.category = "trend"
        data[s.name] = s


def _cpr_offset_and_fill(series_dict, offset, **kwargs):
    """Apply offset and fill to every Series in *series_dict* in-place."""
    for key in list(series_dict):
        series_dict[key] = apply_offset(series_dict[key], offset)
    keys = list(series_dict)
    filled = apply_fill([series_dict[k] for k in keys], **kwargs)
    for i, key in enumerate(keys):
        series_dict[key] = filled[i]


def _cpr_build_dataframe(
    pivot_result, levels, width_analysis, price_position, virgin_cpr
):
    """Build and return the CPR result DataFrame from *pivot_result*."""
    tc, pivot, bc = pivot_result["tc"], pivot_result["pivot"], pivot_result["bc"]
    tc.name, pivot.name, bc.name = "CPR_TC", "CPR_PIVOT", "CPR_BC"
    tc.category = pivot.category = bc.category = "trend"
    data = {tc.name: tc, pivot.name: pivot, bc.name: bc}
    if levels in ["standard", "extended", "all"]:
        for level in ["R1", "R2", "S1", "S2"]:
            _cpr_add_level(pivot_result, data, level)
    if levels in ["extended", "all"]:
        for level in ["R3", "R4", "S3", "S4"]:
            _cpr_add_level(pivot_result, data, level)
    if width_analysis:
        for key, name in [
            ("width", "CPR_WIDTH"),
            ("width_pct", "CPR_WIDTH_PCT"),
            ("width_class", "CPR_WIDTH_CLASS"),
        ]:
            pivot_result[key].name = name
            pivot_result[key].category = "trend"
            data[name] = pivot_result[key]
    if price_position:
        pivot_result["position"].name = "CPR_POSITION"
        pivot_result["position"].category = "trend"
        data["CPR_POSITION"] = pivot_result["position"]
    if virgin_cpr:
        pivot_result["virgin"].name = "CPR_VIRGIN"
        pivot_result["virgin"].category = "trend"
        data["CPR_VIRGIN"] = pivot_result["virgin"]
    cprdf = DataFrame(data)
    cprdf.name = "CPR"
    cprdf.category = "trend"
    return cprdf


def cpr(
    open: Series,
    high: Series,
    low: Series,
    close: Series,
    volume: Optional[Series] = None,
    method: str = "classic",
    timeframe: str = "daily",
    interval: Optional[str] = None,
    levels: str = "standard",
    width_analysis: bool = True,
    price_position: bool = True,
    virgin_cpr: bool = False,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: CPR (Central Pivot Range)"""
    # Validate arguments
    method = method.lower() if isinstance(method, str) else "classic"
    if method not in ["classic", "camarilla", "fibonacci", "woodie"]:
        method = "classic"

    timeframe = timeframe.lower() if isinstance(timeframe, str) else "daily"
    if timeframe not in ["intraday", "daily", "weekly", "monthly"]:
        timeframe = "daily"

    levels = levels.lower() if isinstance(levels, str) else "standard"
    if levels not in ["basic", "standard", "extended", "all"]:
        levels = "standard"

    length = 1  # For verify_series
    open = verify_series(open, length)
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    offset = get_offset(offset)

    if _any_none(open, high, low, close):
        return None

    # Prepare DataFrame for OHLCV processing
    ohlcv_df = DataFrame({"open": open, "high": high, "low": low, "close": close})
    if volume is not None:
        ohlcv_df["volume"] = volume

    # Get previous period OHLCV
    prev_df = get_previous_period_ohlcv(ohlcv_df, timeframe, interval)
    prev_open = prev_df["prev_open"]
    prev_high = prev_df["prev_high"]
    prev_low = prev_df["prev_low"]
    prev_close = prev_df["prev_close"]

    # Calculate Result based on method
    if method == "classic":
        pivot_result = _calculate_classic_pivots(
            prev_high, prev_low, prev_close, levels
        )
    elif method == "camarilla":
        pivot_result = _calculate_camarilla_pivots(
            prev_high, prev_low, prev_close, levels
        )
    elif method == "fibonacci":
        pivot_result = _calculate_fibonacci_pivots(
            prev_high, prev_low, prev_close, levels
        )
    elif method == "woodie":
        pivot_result = _calculate_woodie_pivots(
            prev_high, prev_low, prev_close, prev_open, levels
        )

    tc, pivot, bc = pivot_result["tc"], pivot_result["pivot"], pivot_result["bc"]

    # Optional analysis (stored directly in pivot_result for unified offset/fill)
    if width_analysis:
        (
            pivot_result["width"],
            pivot_result["width_pct"],
            pivot_result["width_class"],
        ) = calculate_cpr_width(
            tc,
            bc,
            pivot,
            narrow_threshold=kwargs.pop("width_narrow", 0.5),
            wide_threshold=kwargs.pop("width_wide", 1.5),
        )
    if price_position:
        pivot_result["position"] = calculate_price_position(close, tc, bc)
    if virgin_cpr:
        pivot_result["virgin"] = detect_virgin_cpr(
            high, low, tc, bc, lookforward=kwargs.pop("virgin_lookforward", 5)
        )

    _cpr_offset_and_fill(pivot_result, offset, **kwargs)
    return _cpr_build_dataframe(
        pivot_result, levels, width_analysis, price_position, virgin_cpr
    )


def _calculate_classic_pivots(
    prev_high: Series, prev_low: Series, prev_close: Series, levels: str = "standard"
) -> dict:
    """Calculate Classic Floor Pivots

    Args:
        prev_high: Previous period high
        prev_low: Previous period low
        prev_close: Previous period close
        levels: 'basic', 'standard', 'extended', or 'all'

    Returns:
        Dict with keys: tc, pivot, bc, r1, r2, r3, r4, s1, s2, s3, s4
    """
    pivot = (prev_high + prev_low + prev_close) / 3
    bc = (prev_high + prev_low) / 2
    tc = (pivot - bc) + pivot

    result = {"tc": tc, "pivot": pivot, "bc": bc}

    if levels in ["standard", "extended", "all"]:
        r1 = (2 * pivot) - prev_low
        s1 = (2 * pivot) - prev_high
        r2 = pivot + (prev_high - prev_low)
        s2 = pivot - (prev_high - prev_low)
        result.update({"r1": r1, "s1": s1, "r2": r2, "s2": s2})

    if levels in ["extended", "all"]:
        r3 = prev_high + 2 * (pivot - prev_low)
        s3 = prev_low - 2 * (prev_high - pivot)
        r4 = prev_high + 3 * (pivot - prev_low)
        s4 = prev_low - 3 * (prev_high - pivot)
        result.update({"r3": r3, "s3": s3, "r4": r4, "s4": s4})

    return result


def _calculate_camarilla_pivots(
    prev_high: Series, prev_low: Series, prev_close: Series, levels: str = "all"
) -> dict:
    """Calculate Camarilla Pivots

    Args:
        prev_high: Previous period high
        prev_low: Previous period low
        prev_close: Previous period close
        levels: 'basic', 'standard', 'extended', or 'all'

    Returns:
        Dict with keys: tc, pivot, bc, r1, r2, r3, r4, s1, s2, s3, s4
    """
    pivot = (prev_high + prev_low + prev_close) / 3
    bc = (prev_high + prev_low) / 2
    tc = (pivot - bc) + pivot

    range_val = prev_high - prev_low

    r1 = prev_close + (range_val * 1.1 / 12)
    r2 = prev_close + (range_val * 1.1 / 6)
    r3 = prev_close + (range_val * 1.1 / 4)
    r4 = prev_close + (range_val * 1.1 / 2)

    s1 = prev_close - (range_val * 1.1 / 12)
    s2 = prev_close - (range_val * 1.1 / 6)
    s3 = prev_close - (range_val * 1.1 / 4)
    s4 = prev_close - (range_val * 1.1 / 2)

    return {
        "tc": tc,
        "pivot": pivot,
        "bc": bc,
        "r1": r1,
        "r2": r2,
        "r3": r3,
        "r4": r4,
        "s1": s1,
        "s2": s2,
        "s3": s3,
        "s4": s4,
    }


def _calculate_fibonacci_pivots(
    prev_high: Series, prev_low: Series, prev_close: Series, levels: str = "all"
) -> dict:
    """Calculate Fibonacci Pivots

    Args:
        prev_high: Previous period high
        prev_low: Previous period low
        prev_close: Previous period close
        levels: 'basic', 'standard', 'extended', or 'all'

    Returns:
        Dict with keys: tc, pivot, bc, r1, r2, r3, s1, s2, s3
    """
    pivot = (prev_high + prev_low + prev_close) / 3
    bc = (prev_high + prev_low) / 2
    tc = (pivot - bc) + pivot

    range_val = prev_high - prev_low

    r1 = pivot + (range_val * 0.382)
    r2 = pivot + (range_val * 0.618)
    r3 = pivot + (range_val * 1.000)

    s1 = pivot - (range_val * 0.382)
    s2 = pivot - (range_val * 0.618)
    s3 = pivot - (range_val * 1.000)

    return {
        "tc": tc,
        "pivot": pivot,
        "bc": bc,
        "r1": r1,
        "r2": r2,
        "r3": r3,
        "s1": s1,
        "s2": s2,
        "s3": s3,
    }


def _calculate_woodie_pivots(
    prev_high: Series,
    prev_low: Series,
    prev_close: Series,
    prev_open: Series,
    levels: str = "standard",
) -> dict:
    """Calculate Woodie's Pivots

    Args:
        prev_high: Previous period high
        prev_low: Previous period low
        prev_close: Previous period close
        prev_open: Previous period open
        levels: 'basic', 'standard', 'extended', or 'all'

    Returns:
        Dict with keys: tc, pivot, bc, r1, r2, s1, s2
    """
    pivot = (prev_high + prev_low + 2 * prev_close) / 4
    bc = (prev_high + prev_low) / 2
    tc = (pivot - bc) + pivot

    result = {"tc": tc, "pivot": pivot, "bc": bc}

    if levels in ["standard", "extended", "all"]:
        r1 = (2 * pivot) - prev_low
        s1 = (2 * pivot) - prev_high
        r2 = pivot + (prev_high - prev_low)
        s2 = pivot - (prev_high - prev_low)
        result.update({"r1": r1, "s1": s1, "r2": r2, "s2": s2})

    return result


cpr.__doc__ = """CPR (Central Pivot Range)

Central Pivot Range is a trending indicator that helps identify potential
support and resistance levels for the trading session based on previous
period's price action.

Sources:
    https://tradingqna.com/what-is-central-pivot-range-cpr-how-to-trade-using-it/
    https://www.incrediblecharts.com/indicators/pivot_point_calculator.php

Examples:
    import pandas_ta_classic as ta
    df.ta.cpr(method='classic', timeframe='daily', levels='standard', append=True)
    df.ta.cpr(method='camarilla', levels='all', virgin_cpr=True, append=True)

    # Intraday CPR
    df.ta.cpr(method='classic', timeframe='intraday', interval='5min', append=True)

    # Direct function call
    result = ta.cpr(df['open'], df['high'], df['low'], df['close'],
                    method='fibonacci', levels='extended')

Calculation:
    Default Inputs:
        method="classic", timeframe="daily", levels="standard"

    Classic Floor Pivots:
        Pivot = (H + L + C) / 3
        BC = (H + L) / 2
        TC = (Pivot - BC) + Pivot
        R1 = (2 * Pivot) - L
        S1 = (2 * Pivot) - H
        R2 = Pivot + (H - L)
        S2 = Pivot - (H - L)
        R3 = H + 2 * (Pivot - L)
        S3 = L - 2 * (H - Pivot)
        R4 = H + 3 * (Pivot - L)
        S4 = L - 3 * (H - Pivot)

    Camarilla Pivots:
        Pivot = (H + L + C) / 3
        BC = (H + L) / 2
        TC = (Pivot - BC) + Pivot
        Range = H - L
        R1 = C + (Range * 1.1/12)
        R2 = C + (Range * 1.1/6)
        R3 = C + (Range * 1.1/4)
        R4 = C + (Range * 1.1/2)
        S1 = C - (Range * 1.1/12)
        S2 = C - (Range * 1.1/6)
        S3 = C - (Range * 1.1/4)
        S4 = C - (Range * 1.1/2)

    Fibonacci Pivots:
        Pivot = (H + L + C) / 3
        BC = (H + L) / 2
        TC = (Pivot - BC) + Pivot
        Range = H - L
        R1 = Pivot + (Range * 0.382)
        R2 = Pivot + (Range * 0.618)
        R3 = Pivot + (Range * 1.000)
        S1 = Pivot - (Range * 0.382)
        S2 = Pivot - (Range * 0.618)
        S3 = Pivot - (Range * 1.000)

    Woodie's Pivots:
        Pivot = (H + L + 2*C) / 4
        BC = (H + L) / 2
        TC = (Pivot - BC) + Pivot
        R1 = (2 * Pivot) - L
        S1 = (2 * Pivot) - H
        R2 = Pivot + (H - L)
        S2 = Pivot - (H - L)

Args:
    open (pd.Series): Series of 'open's
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    volume (pd.Series, optional): Series of 'volume's
    method (str): Pivot calculation method.
        Options: 'classic', 'camarilla', 'fibonacci', 'woodie'. Default: 'classic'
    timeframe (str): Time context for CPR calculation.
        Options: 'intraday', 'daily', 'weekly', 'monthly'. Default: 'daily'
    interval (str, optional): For intraday only - data interval.
        Examples: '1min', '5min', '15min', '30min', '1H'. Default: None
    levels (str): Which pivot levels to calculate.
        Options: 'basic' (TC/P/BC), 'standard' (+R1/R2/S1/S2),
                 'extended' (+R3/R4/S3/S4), 'all'. Default: 'standard'
    width_analysis (bool): Calculate CPR width metrics. Default: True
    price_position (bool): Calculate price position relative to CPR. Default: True
    virgin_cpr (bool): Detect virgin (untested) CPR levels. Default: False
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    width_narrow (float): Threshold for narrow CPR classification (%). Default: 0.5
    width_wide (float): Threshold for wide CPR classification (%). Default: 1.5
    virgin_lookforward (int): Periods to look ahead for virgin CPR detection. Default: 5
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: CPR levels and analysis columns.
        Core columns: CPR_TC, CPR_PIVOT, CPR_BC
        S/R columns (depends on 'levels'): CPR_R1, CPR_R2, CPR_S1, CPR_S2, etc.
        Analysis columns: CPR_WIDTH, CPR_WIDTH_PCT, CPR_WIDTH_CLASS, CPR_POSITION, CPR_VIRGIN
"""
