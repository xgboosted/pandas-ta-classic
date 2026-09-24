# CPR Utility Functions

import numpy as np
from pandas import DataFrame, Series


def _prev_period_ohlcv(result: DataFrame, df: DataFrame, freq: str) -> None:
    """Assign prev_* columns from the last completed calendar period before each bar's own.

    Bars are grouped by the calendar period they fall in (``to_period``), each
    observed period is aggregated, and every bar reads the aggregate of the
    observed period before its own. A bar never sees its own period, so the
    levels are causal whatever the bar frequency, and a period with no bars
    (a weekend, a holiday week) is skipped rather than read as NaN.
    """
    wall_clock = df.index.tz_localize(None) if df.index.tz is not None else df.index  # periods follow local dates
    key = wall_clock.to_period(freq)
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    prev = df.groupby(key).agg(agg).shift(1)
    for col in agg:
        result[f"prev_{col}"] = prev[col].reindex(key).to_numpy()


def get_previous_period_ohlcv(df: DataFrame, timeframe: str = "daily", interval: str | None = None) -> DataFrame:
    """Get previous period OHLCV data

    For daily: the previous bar.
    For intraday/weekly/monthly: the last completed calendar day, week
    (Monday to Sunday) or month before the bar's own.

    Args:
        df: DataFrame with OHLCV data and datetime index
        timeframe: 'intraday', 'daily', 'weekly', 'monthly'
        interval: For intraday - the data interval ('1min', '5min', etc.)

    Returns:
        DataFrame with columns: prev_open, prev_high, prev_low, prev_close, prev_volume
    """
    result = df.copy()

    if timeframe == "daily":
        result["prev_open"] = df["open"].shift(1)
        result["prev_high"] = df["high"].shift(1)
        result["prev_low"] = df["low"].shift(1)
        result["prev_close"] = df["close"].shift(1)
        if "volume" in df.columns:
            result["prev_volume"] = df["volume"].shift(1)
    else:
        # resample() labels weekly and monthly bins at the period end, so a bar
        # on that label date read its own period (look-ahead), and shifting the
        # bins to avoid that lagged every other bar by two periods.
        _prev_period_ohlcv(result, df, {"intraday": "D", "weekly": "W", "monthly": "M"}[timeframe])

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

    Not causal: the flag at bar ``i`` is decided by bars ``i + 1 .. i + lookforward``,
    so it is unavailable in real time and must not gate backtest entries.

    Args:
        high: High price series
        low: Low price series
        tc: Top Central series
        bc: Bottom Central series
        lookforward: Number of periods ahead to check if CPR was tested (default 5)

    Returns:
        Boolean Series: True if CPR remains untested (virgin) in the lookforward period
    """
    # For each forward offset k, mark bars whose price range at i + k falls
    # inside the CPR range at i. A bar is touched when any offset does; a
    # virgin level is one the next `lookforward` bars never touch.
    touched = Series(False, index=high.index)
    for k in range(1, lookforward + 1):
        touched |= (high.shift(-k) >= bc) & (low.shift(-k) <= tc)
    virgin = (bc.notna() & tc.notna()) & ~touched
    if lookforward:
        # The last `lookforward` bars have no future to look into: the shifted
        # comparisons leave `touched` False there, so `~touched` would mark
        # them virgin. Force them False instead, matching the original loop,
        # which never reached those bars.
        virgin.iloc[-lookforward:] = False

    return virgin
