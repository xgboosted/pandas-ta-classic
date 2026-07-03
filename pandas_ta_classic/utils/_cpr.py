# CPR Utility Functions
from typing import Optional

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


def _assign_prev_ohlcv(result: DataFrame, df: DataFrame, prev: DataFrame) -> None:
    """Reindex *prev* onto *result* index (ffill) and assign prev_* columns in-place."""
    for col in ("open", "high", "low", "close"):
        result[f"prev_{col}"] = prev[col].reindex(result.index, method="ffill")
    if "volume" in df.columns:
        result["prev_volume"] = prev["volume"].reindex(result.index, method="ffill")


def _resample_ohlcv(df: DataFrame, rule: str) -> DataFrame:
    """Resample OHLCV *df* to *rule* frequency."""
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in df.columns:
        agg["volume"] = "sum"
    return df.resample(rule).agg(agg)


def get_previous_period_ohlcv(df: DataFrame, timeframe: str = "daily", interval: Optional[str] = None) -> DataFrame:
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
        prev_daily = _resample_ohlcv(df, "D").shift(1)
        _assign_prev_ohlcv(result, df, prev_daily)

    elif timeframe == "daily":
        result["prev_open"] = df["open"].shift(1)
        result["prev_high"] = df["high"].shift(1)
        result["prev_low"] = df["low"].shift(1)
        result["prev_close"] = df["close"].shift(1)
        if "volume" in df.columns:
            result["prev_volume"] = df["volume"].shift(1)

    elif timeframe == "weekly":
        prev_weekly = _resample_ohlcv(df, "W").shift(1)
        _assign_prev_ohlcv(result, df, prev_weekly)

    elif timeframe == "monthly":
        prev_monthly = _resample_ohlcv(df, "M").shift(1)
        _assign_prev_ohlcv(result, df, prev_monthly)

    return result


def calculate_cpr_width(
    tc: Series,
    bc: Series,
    pivot: Series,
    narrow_threshold: float = 0.5,
    wide_threshold: float = 1.5,
) -> tuple[Series, Series, Series]:
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
            width_class: Integer classification: -1 (narrow), 0 (medium), 1 (wide)
    """
    width = tc - bc
    width_pct = (width / pivot) * 100

    width_class = Series(0, index=width.index, dtype=np.int8)
    width_class[width_pct < narrow_threshold] = -1
    width_class[width_pct > wide_threshold] = 1

    return width, width_pct, width_class


def calculate_price_position(close: Series, tc: Series, bc: Series) -> Series:
    """Determine price position relative to CPR

    Args:
        close: Close price series
        tc: Top Central series
        bc: Bottom Central series

    Returns:
        Series with integer values: 1 (above TC), 0 (inside CPR), -1 (below BC)
    """
    position = Series(0, index=close.index, dtype=np.int8)
    position[close > tc] = 1
    position[close < bc] = -1
    return position


def detect_virgin_cpr(high: Series, low: Series, tc: Series, bc: Series, lookforward: int = 5) -> Series:
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
