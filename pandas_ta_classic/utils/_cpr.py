# -*- coding: utf-8 -*-
# CPR Utility Functions
from typing import Optional, Tuple

import pandas as pd
from pandas import DataFrame, Series


def get_previous_period_ohlcv(
    df: DataFrame, timeframe: str = "daily", interval: Optional[str] = None
) -> DataFrame:
    """Get previous period OHLCV data using resample + shift

    For intraday: Resamples to daily, shifts by 1 day, forward fills
    For daily/weekly/monthly: Simple shift or resample as appropriate

    Args:
        df: DataFrame with OHLCV data and datetime index
        timeframe: 'intraday', 'daily', 'weekly', 'monthly'
        interval: For intraday - the data interval ('1min', '5min', etc.)

    Returns:
        DataFrame with columns: prev_open, prev_high, prev_low, prev_close, prev_volume
    """
    result = df.copy()

    if timeframe == "intraday":
        # Resample to daily
        daily = df.resample("D").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum" if "volume" in df.columns else "last",
            }
        )

        # Shift by 1 day
        prev_daily = daily.shift(1)

        # Forward fill to all intraday bars
        result["prev_open"] = prev_daily["open"].reindex(result.index, method="ffill")
        result["prev_high"] = prev_daily["high"].reindex(result.index, method="ffill")
        result["prev_low"] = prev_daily["low"].reindex(result.index, method="ffill")
        result["prev_close"] = prev_daily["close"].reindex(result.index, method="ffill")
        if "volume" in df.columns:
            result["prev_volume"] = prev_daily["volume"].reindex(
                result.index, method="ffill"
            )

    elif timeframe == "daily":
        # Simple shift by 1 period
        result["prev_open"] = df["open"].shift(1)
        result["prev_high"] = df["high"].shift(1)
        result["prev_low"] = df["low"].shift(1)
        result["prev_close"] = df["close"].shift(1)
        if "volume" in df.columns:
            result["prev_volume"] = df["volume"].shift(1)

    elif timeframe == "weekly":
        # Resample to weekly, shift by 1 week
        weekly = df.resample("W").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum" if "volume" in df.columns else "last",
            }
        )
        prev_weekly = weekly.shift(1)

        result["prev_open"] = prev_weekly["open"].reindex(result.index, method="ffill")
        result["prev_high"] = prev_weekly["high"].reindex(result.index, method="ffill")
        result["prev_low"] = prev_weekly["low"].reindex(result.index, method="ffill")
        result["prev_close"] = prev_weekly["close"].reindex(
            result.index, method="ffill"
        )
        if "volume" in df.columns:
            result["prev_volume"] = prev_weekly["volume"].reindex(
                result.index, method="ffill"
            )

    elif timeframe == "monthly":
        # Resample to monthly, shift by 1 month
        monthly = df.resample("M").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum" if "volume" in df.columns else "last",
            }
        )
        prev_monthly = monthly.shift(1)

        result["prev_open"] = prev_monthly["open"].reindex(result.index, method="ffill")
        result["prev_high"] = prev_monthly["high"].reindex(result.index, method="ffill")
        result["prev_low"] = prev_monthly["low"].reindex(result.index, method="ffill")
        result["prev_close"] = prev_monthly["close"].reindex(
            result.index, method="ffill"
        )
        if "volume" in df.columns:
            result["prev_volume"] = prev_monthly["volume"].reindex(
                result.index, method="ffill"
            )

    return result


def calculate_cpr_width(
    tc: Series,
    bc: Series,
    pivot: Series,
    narrow_threshold: float = 0.5,
    wide_threshold: float = 1.5,
) -> Tuple[Series, Series, Series]:
    """Calculate CPR width metrics

    Args:
        tc: Top Central series
        bc: Bottom Central series
        pivot: Pivot series
        narrow_threshold: Percentage threshold for narrow CPR (default 0.5%)
        wide_threshold: Percentage threshold for wide CPR (default 1.5%)

    Returns:
        Tuple of (width, width_pct, width_class):
            width: Absolute width (TC - BC)
            width_pct: Percentage width relative to Pivot
            width_class: Classification ('narrow', 'medium', 'wide')
    """
    width = tc - bc
    width_pct = (width / pivot) * 100

    # Classify width
    width_class = Series("medium", index=width.index)
    width_class[width_pct < narrow_threshold] = "narrow"
    width_class[width_pct > wide_threshold] = "wide"

    return width, width_pct, width_class


def calculate_price_position(close: Series, tc: Series, bc: Series) -> Series:
    """Determine price position relative to CPR

    Args:
        close: Close price series
        tc: Top Central series
        bc: Bottom Central series

    Returns:
        Series with values: 'above_tc', 'inside_cpr', 'below_bc'
    """
    position = Series("inside_cpr", index=close.index)
    position[close > tc] = "above_tc"
    position[close < bc] = "below_bc"
    return position


def round_to_strike(price: Series, round_to: int = 50) -> Series:
    """Round prices to nearest strike interval

    Args:
        price: Price series
        round_to: Strike interval (50, 100, etc.)

    Returns:
        Rounded strike prices
    """
    return (price / round_to).round() * round_to


def calculate_option_strikes(
    tc: Series, pivot: Series, bc: Series, strike_round: int = 50
) -> dict:
    """Calculate option strike recommendations based on CPR levels

    Args:
        tc: Top Central series
        pivot: Pivot series
        bc: Bottom Central series
        strike_round: Strike interval to round to

    Returns:
        Dict with keys: tc_strike, pivot_strike, bc_strike,
                       call_otm1, call_otm2, put_otm1, put_otm2
    """
    tc_strike = round_to_strike(tc, strike_round)
    pivot_strike = round_to_strike(pivot, strike_round)
    bc_strike = round_to_strike(bc, strike_round)

    call_otm1 = tc_strike + strike_round
    call_otm2 = tc_strike + (strike_round * 2)

    put_otm1 = bc_strike - strike_round
    put_otm2 = bc_strike - (strike_round * 2)

    return {
        "tc_strike": tc_strike,
        "pivot_strike": pivot_strike,
        "bc_strike": bc_strike,
        "call_otm1": call_otm1,
        "call_otm2": call_otm2,
        "put_otm1": put_otm1,
        "put_otm2": put_otm2,
    }


def detect_cpr_breakout(
    close: Series, high: Series, low: Series, tc: Series, bc: Series, lookback: int = 1
) -> Tuple[Series, Series]:
    """Detect CPR breakouts for option entry signals

    Args:
        close: Close price series
        high: High price series
        low: Low price series
        tc: Top Central series
        bc: Bottom Central series
        lookback: Periods to look back for breakout detection

    Returns:
        Tuple of (breakout_call, breakout_put):
            breakout_call: Boolean - Bullish breakout above TC
            breakout_put: Boolean - Bearish breakout below BC
    """
    # Bullish breakout: Price crosses above TC
    was_below_tc = close.shift(lookback) <= tc.shift(lookback)
    now_above_tc = close > tc
    breakout_call = was_below_tc & now_above_tc

    # Bearish breakout: Price crosses below BC
    was_above_bc = close.shift(lookback) >= bc.shift(lookback)
    now_below_bc = close < bc
    breakout_put = was_above_bc & now_below_bc

    return breakout_call, breakout_put


def detect_cpr_rejection(
    close: Series,
    high: Series,
    low: Series,
    tc: Series,
    bc: Series,
    threshold: float = 0.2,
) -> Tuple[Series, Series]:
    """Detect CPR rejections for reversal option trades

    Args:
        close: Close price series
        high: High price series
        low: Low price series
        tc: Top Central series
        bc: Bottom Central series
        threshold: Percentage tolerance for "touching" the level

    Returns:
        Tuple of (rejection_call, rejection_put):
            rejection_call: Boolean - Bullish rejection from BC
            rejection_put: Boolean - Bearish rejection from TC
    """
    # Bullish rejection: Low touches BC, close above BC
    touched_bc = low <= (bc * (1 + threshold / 100))
    closed_above_bc = close > bc
    rejection_call = touched_bc & closed_above_bc

    # Bearish rejection: High touches TC, close below TC
    touched_tc = high >= (tc * (1 - threshold / 100))
    closed_below_tc = close < tc
    rejection_put = touched_tc & closed_below_tc

    return rejection_call, rejection_put


def detect_virgin_cpr(
    high: Series, low: Series, tc: Series, bc: Series, lookforward: int = 5
) -> Series:
    """Detect Virgin CPR levels (untested CPR ranges)

    A Virgin CPR is one where price has not entered the CPR range (between TC and BC)
    in the next N periods after the CPR was formed. These levels often act as
    strong support/resistance when tested later.

    Args:
        high: High price series
        low: Low price series
        tc: Top Central series
        bc: Bottom Central series
        lookforward: Number of periods ahead to check if CPR was tested (default 5)

    Returns:
        Boolean Series: True if CPR remains untested (virgin) in the lookforward period
    """
    virgin = Series(False, index=high.index)

    # For each bar, check if price touches CPR in next 'lookforward' periods
    for i in range(len(high) - lookforward):
        # Get CPR levels for current period
        cpr_tc = tc.iloc[i]
        cpr_bc = bc.iloc[i]

        # Skip if CPR values are NaN
        if pd.isna(cpr_tc) or pd.isna(cpr_bc):
            continue

        # Check if price touched CPR in next 'lookforward' periods
        # CPR is touched if high > bc AND low < tc (price entered the range)
        future_highs = high.iloc[i + 1 : i + 1 + lookforward]
        future_lows = low.iloc[i + 1 : i + 1 + lookforward]

        # CPR is virgin if price never entered the CPR range
        # Price enters CPR if: (high >= bc) AND (low <= tc)
        cpr_touched = ((future_highs >= cpr_bc) & (future_lows <= cpr_tc)).any()

        # Virgin CPR = NOT touched
        virgin.iloc[i] = not cpr_touched

    return virgin
