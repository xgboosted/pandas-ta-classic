import contextvars
import functools
import inspect
import logging
import math
import sys
import warnings
from collections.abc import Callable
from numbers import Real
from sys import float_info as sflt
from typing import Any, TypeGuard

import numpy as np
from pandas import DataFrame, DatetimeIndex, RangeIndex, Series, date_range
from pandas.api.types import is_datetime64_any_dtype

logger = logging.getLogger(__name__)


def _validate_number(val: Any, default: Any, name: str, *, integer: bool, gt: float | None, ge: float | None, lt: float | None) -> Any:
    """Shared body of :func:`_pos_int` and :func:`_pos_float`.

    ``None`` means "not given" and returns *default*. Anything else must be a
    finite real number (not a bool) inside the bounds, and a whole number when
    *integer* is set; otherwise ``ValueError`` names the indicator, the
    parameter and the value instead of silently substituting the default.
    """
    if val is None:
        return default
    valid = isinstance(val, Real) and not isinstance(val, bool) and math.isfinite(val)
    valid = valid and (not integer or float(val).is_integer())
    valid = valid and (gt is None or val > gt) and (ge is None or val >= ge) and (lt is None or val < lt)
    if not valid:
        indicator = sys._getframe(2).f_code.co_name  # the indicator that called _pos_int/_pos_float
        bounds = " and ".join(f"{op} {bound}" for op, bound in ((">", gt), (">=", ge), ("<", lt)) if bound is not None)
        kind = "an integer" if integer else "a number"
        requirement = f"{kind} {bounds}" if bounds else kind
        raise ValueError(f"{indicator}() {name} must be {requirement}, got {val!r}")
    return int(val) if integer else float(val)


def _pos_int(val: Any, default: Any, name: str = "value", *, gt: float | None = 0, ge: float | None = None, lt: float | None = None) -> Any:
    """Return ``int(val)``, *default* when *val* is None, or raise ValueError.

    The bound defaults to ``> 0``; pass ``ge`` (with ``gt=None``) or ``lt`` for others.
    """
    return _validate_number(val, default, name, integer=True, gt=gt, ge=ge, lt=lt)


def _pos_float(val: Any, default: Any, name: str = "value", *, gt: float | None = 0, ge: float | None = None, lt: float | None = None) -> Any:
    """Return ``float(val)``, *default* when *val* is None, or raise ValueError."""
    return _validate_number(val, default, name, integer=False, gt=gt, ge=ge, lt=lt)


def _number(val: Any, default: Any, name: str = "value", *, gt: float | None = None, ge: float | None = None, lt: float | None = None) -> Any:
    """Like :func:`_pos_float` but unbounded by default: any finite number, including 0 and negatives."""
    return _validate_number(val, default, name, integer=False, gt=gt, ge=ge, lt=lt)


def apply_offset(
    series: Series | DataFrame | list[Series | DataFrame],
    offset: int = 0,
) -> Series | DataFrame | list[Series | DataFrame]:
    """Shift one or more Series/DataFrames by *offset* periods.

    Args:
        series: A single Series/DataFrame, or a list of them.
        offset: Number of periods to shift. ``0`` means no shift.

    Returns:
        The shifted object(s), same type structure as *series*.
    """
    if isinstance(series, (list, tuple)):
        return [apply_offset(s, offset) for s in series]
    return series.shift(offset) if offset != 0 else series


def apply_fill(
    series: Series | DataFrame | list[Series | DataFrame],
    **kwargs: Any,
) -> Series | DataFrame | list[Series | DataFrame]:
    """Apply fillna and fill_method from kwargs to one or more Series/DataFrames.

    Args:
        series: A single Series/DataFrame, or a list of them.
        **kwargs: Recognised keys:
            ``fillna`` -- value passed to ``Series.fillna()``.
            ``fill_method`` -- ``"ffill"`` or ``"bfill"``.

    Returns:
        The processed object(s), same type structure as *series*.
    """
    if isinstance(series, (list, tuple)):
        return [apply_fill(s, **kwargs) for s in series]
    if "fillna" in kwargs:
        series.fillna(kwargs["fillna"], inplace=True)
    fill_method = kwargs.get("fill_method")
    if fill_method == "ffill":
        series.ffill(inplace=True)
    elif fill_method == "bfill":
        series.bfill(inplace=True)
    return series


def _bool_param(val: Any, default: bool, name: str) -> bool:
    """Return *val* for a bool (numpy bools included), *default* for None; raise ValueError otherwise.

    ``bool(x) if isinstance(x, bool) else default`` turned ``talib=1`` into
    False and ``asint=0`` into True without a word.
    """
    if val is None:
        return default
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    indicator = sys._getframe(1).f_code.co_name
    raise ValueError(f"{indicator}() {name} must be True or False, got {val!r}")


def _str_param(val: Any, default: str, name: str, *, choices: Any = None, lower: bool = True) -> str:
    """Return a (lower-cased) string, *default* for None; raise ValueError for other types or unknown choices."""
    indicator = sys._getframe(1).f_code.co_name
    if val is None:
        return default
    if not isinstance(val, str) or not val:
        raise ValueError(f"{indicator}() {name} must be a non-empty string, got {val!r}")
    out = val.lower() if lower else val
    if choices is not None and out not in choices:
        raise ValueError(f"{indicator}() {name} must be one of {sorted(choices)}, got {val!r}")
    return out


def get_drift(x: int | None) -> int:
    """Return *x* as a positive int, 1 when None; raise ValueError otherwise.

    A zero or negative drift used to become 1 silently; a negative drift would
    difference against future bars.
    """
    return _validate_number(x, 1, "drift", integer=True, gt=0, ge=None, lt=None)


def get_offset(x: int | None) -> int:
    """Return *x* as an int (negative allowed), 0 when None; raise ValueError otherwise."""
    return _validate_number(x, 0, "offset", integer=True, gt=None, ge=None, lt=None)


def is_datetime_ordered(df: DataFrame | Series) -> bool:
    """Returns True if the index is a datetime and ordered."""
    index_is_datetime = is_datetime64_any_dtype(df.index)
    if not index_is_datetime or len(df.index) < 2:
        return False
    try:
        return bool(df.index[0] < df.index[-1])
    except (IndexError, TypeError):
        return False


def is_percent(x: float | None) -> TypeGuard[float]:
    if isinstance(x, (int, float)):
        return x is not None and x >= 0 and x <= 100
    return False


def leading_nan_rows(*series: Series) -> int:
    """Number of leading rows before every series has a finite value."""
    finite = np.isfinite(np.vstack([s.to_numpy(dtype=float) for s in series])).all(axis=0)
    hits = np.flatnonzero(finite)
    return int(hits[0]) if hits.size else finite.size


def skip_leading_nan(*names: str) -> Callable:
    """Compute an indicator on the rows after a leading NaN run, then pad back.

    Chained input (another indicator's output) always starts with NaN. A
    recursion seeded from the first bar carries that NaN forward forever, so
    the whole result came back NaN. The decorated function instead receives
    every Series argument trimmed to start at the first row where all of
    *names* are finite; its result is reindexed to the original index, with
    ``name`` and ``category`` preserved. Input without a leading NaN run, or
    that is entirely NaN, is passed through unchanged.
    """

    def decorator(fn: Callable) -> Callable:
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            bound = sig.bind_partial(*args, **kwargs)
            primary = [bound.arguments.get(n) for n in names]
            if not all(isinstance(s, Series) for s in primary) or len({s.size for s in primary}) != 1:
                return fn(*args, **kwargs)
            size = primary[0].size
            start = leading_nan_rows(*primary)
            if not 0 < start < size:
                return fn(*args, **kwargs)
            for key, value in bound.arguments.items():
                if isinstance(value, Series) and value.size == size:
                    bound.arguments[key] = value.iloc[start:]
            result = fn(*bound.args, **bound.kwargs)
            if result is None:
                return None
            padded = result.reindex(primary[0].index)
            for attr in ("name", "category"):
                if hasattr(result, attr):
                    setattr(padded, attr, getattr(result, attr))
            return padded

        return wrapper

    return decorator


# Nesting depth of nan_on_short_input calls: only the outermost indicator call
# converts a short-input None into an all-NaN result, so indicators calling
# each other internally keep seeing None and their early returns still work.
_INDICATOR_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar("pandas_ta_classic_indicator_depth", default=0)
_PROBE_ROWS = 1000
_PROBE_MAX_ROWS = 50_000
_PROBE_HARD_MAX_ROWS = 2_000_000


def _probe_inputs(arguments: dict, rows: int) -> dict:
    """Replace every Series argument with a long, well-formed synthetic series.

    Values are deterministic OHLCV-shaped data on the same index type as the
    caller's first Series, so every indicator can run on them.
    """
    first = next(v for v in arguments.values() if isinstance(v, Series))
    if isinstance(first.index, DatetimeIndex):
        end = first.index[-1] if len(first.index) else "2024-01-01"
        # minute steps: 2M rows span under 4 years, well inside the datetime64[ns] range
        index = date_range(end=end, periods=rows, freq="min", tz=first.index.tz)
    else:
        index = RangeIndex(rows)
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, rows))
    open_ = np.concatenate(([close[0]], close[:-1]))
    columns = {
        "open_": open_,
        "high": np.maximum(open_, close) + rng.random(rows),
        "low": np.minimum(open_, close) - rng.random(rows),
        "volume": rng.integers(1_000, 100_000, rows).astype(float),
        "periods": np.full(rows, 10.0),
        "trend": (np.arange(rows) // 20 % 2).astype(int),
    }
    # keep each name: some outputs are named after their inputs (vp: low_close, pos_volume)
    return {
        key: (Series(columns.get(key, close), index=index, name=value.name) if isinstance(value, Series) else value)
        for key, value in arguments.items()
    }


def _nan_like(template: Any, index: Any, rows: int) -> Any:
    """All-NaN copy of *template*'s structure (name, columns, category) on *index*."""
    target = index if len(template) == rows else template.index  # non-time-series output (vp bins) keeps its own rows
    if isinstance(template, DataFrame):
        out: Any = DataFrame(np.nan, index=target, columns=template.columns, dtype=float)
    else:
        out = Series(np.nan, index=target, name=template.name, dtype=float)
    for attr in ("name", "category"):
        if hasattr(template, attr):
            setattr(out, attr, getattr(template, attr))
    return out


def _require_shared_index(indicator: str, sig: inspect.Signature, args: tuple, kwargs: dict) -> None:
    """Raise ValueError when two non-empty Series inputs have no index label in common.

    Inputs of different lengths are fine: pandas aligns them on the labels they
    share (a benchmark with a longer history, a volume series that starts later).
    Inputs with no label in common -- a RangeIndex next to a DatetimeIndex, a
    tz-aware next to a naive index, two date ranges that never overlap -- can
    only produce an all-NaN result on the union of both indexes, twice as long
    as either input, so they are a caller error.
    """
    series = [v for v in (*args, *kwargs.values()) if isinstance(v, Series) and not v.empty]
    for other in series[1:]:
        first = series[0]
        if other.index is first.index or other.index.equals(first.index):
            continue
        if first.index.intersection(other.index).empty:
            names = {id(v): k for k, v in sig.bind(*args, **kwargs).arguments.items()}
            raise ValueError(
                f"{indicator}() inputs share no index labels: {names.get(id(first), '?')} has {_describe_index(first.index)}, "
                f"{names.get(id(other), '?')} has {_describe_index(other.index)}. Align them first, e.g. by taking both from one DataFrame."
            )


def _describe_index(index: Any) -> str:
    tz = getattr(index, "tz", None)
    return f"{type(index).__name__}{f' ({tz})' if tz is not None else ''} {index[0]!r} .. {index[-1]!r}"


def nan_on_short_input(fn: Callable) -> Callable:
    """Return an all-NaN result instead of None when the input is shorter than the window (issue #145, case B).

    ``close.rolling(50).mean()`` on 10 rows returns 10 NaNs; an indicator did
    the same job by returning None, which silently dropped the column in
    ``df.ta.<indicator>(append=True)`` and turned into someone else's error
    downstream. When the outermost call returns None although Series were
    passed, the call is repeated on long synthetic data of the same shape to
    learn the output's name, columns and category, and an all-NaN result with
    the caller's index is returned. An empty Series skips the first call and
    gets the same all-NaN (zero-row) result. A None that the synthetic call
    reproduces (a required input missing) stays None.
    """
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if _INDICATOR_DEPTH.get():
            return fn(*args, **kwargs)
        token = _INDICATOR_DEPTH.set(1)
        try:
            # An empty Series is shorter than any window, but indicators that read
            # bar 0 (obv, psar, hwma, ...) raised IndexError on it instead of
            # returning None, so go straight to the all-NaN result.
            empty = any(isinstance(v, Series) and v.empty for v in (*args, *kwargs.values()))
            if empty:
                logger.warning(f"[X] Series has 0 rows; {fn.__name__}() result is all NaN.")
            else:
                _require_shared_index(fn.__name__, sig, args, kwargs)
                result = fn(*args, **kwargs)
                if result is not None:
                    return result
            bound = sig.bind(*args, **kwargs)
            series = {k: v for k, v in bound.arguments.items() if isinstance(v, Series)}
            if not series:
                return None
            # results align to close, like a normal result; otherwise to the first Series
            first = series.get("close", next(iter(series.values())))
            numbers = [
                v
                for v in list(bound.arguments.values()) + list(bound.arguments.get("kwargs", {}).values())
                if isinstance(v, Real) and not isinstance(v, bool)
            ]
            # Long enough for any window the caller asked for. The first try is capped so a large
            # non-window number (eom's divisor) cannot request millions of rows; only when that is
            # still too short for the window (sma(length=60000)) is it retried at the full length.
            needed = max([_PROBE_ROWS] + [int(3 * abs(v)) + 10 for v in numbers if math.isfinite(v)])
            original = dict(bound.arguments)
            template = None
            # ponytail: windows past _PROBE_HARD_MAX_ROWS / 3 still come back None; raise the limit if one matters
            for rows in dict.fromkeys((min(_PROBE_MAX_ROWS, needed), min(needed, _PROBE_HARD_MAX_ROWS))):
                bound.arguments.update(_probe_inputs(original, rows))
                with warnings.catch_warnings():  # the caller's call already warned; the probe must not repeat it
                    warnings.simplefilter("ignore")
                    template = fn(*bound.args, **bound.kwargs)
                if template is not None:
                    break
            if template is None or isinstance(template, tuple):
                return fn(*args, **kwargs) if empty else None
            return _nan_like(template, first.index, rows)
        finally:
            _INDICATOR_DEPTH.reset(token)

    return wrapper


def non_zero_range(high: Series, low: Series) -> Series:
    """Returns the difference of two series, replacing exact zeros with epsilon.  This occurs commonly in crypto data when 'high' = 'low'.

    The replacement is pointwise: a flat bar at row ``t`` only affects row ``t``.
    Rows with a non-zero range keep their exact difference, so the value at row
    ``t`` never depends on bars at ``t + 1`` or later.
    """
    diff = high - low
    return diff.where(diff != 0, sflt.epsilon)


def recent_maximum_index(x: Series) -> int:
    return int(np.argmax(x[::-1]))


def recent_minimum_index(x: Series) -> int:
    return int(np.argmin(x[::-1]))


def signed_series(series: Series, initial: int | None = None) -> Series:
    """Returns a Signed Series with or without an initial value

    Default Example:
    series = Series([3, 2, 2, 1, 1, 5, 6, 6, 7, 5])
    and returns:
    sign = Series([NaN, -1.0, 0.0, -1.0, 0.0, 1.0, 1.0, 0.0, 1.0, -1.0])
    """
    series = verify_series(series)
    sign = series.diff(1)
    sign[sign > 0] = 1
    sign[sign < 0] = -1
    sign.iloc[0] = initial
    return sign


# TA-Lib MA_Type enum values are frozen ABI constants, so they are mapped
# directly here rather than importing talib just to read them. This keeps
# tal_ma usable (and testable) without the optional talib dependency.
_TAL_MA_TYPES = {
    "sma": 0,
    "ema": 1,
    "wma": 2,
    "dema": 3,
    "tema": 4,
    "trima": 5,
    "kama": 6,
    "mama": 7,
    "t3": 8,
}


def tal_ma(name: str) -> int:
    """Return the TA-Lib MA_Type enum value for an MA name (``sma``..``t3``).

    Raises:
        TypeError: if *name* is not a string.
        ValueError: if *name* is not a recognised TA-Lib MA type.
    """
    if not isinstance(name, str):
        raise TypeError(f"tal_ma expects a str MA name, got {type(name).__name__}")
    key = name.lower()
    if key not in _TAL_MA_TYPES:
        raise ValueError(f"Unknown TA-Lib MA type {name!r}; valid: {sorted(_TAL_MA_TYPES)}")
    return _TAL_MA_TYPES[key]


def unsigned_differences(series: Series, amount: int | None = None, *, asint: bool = False) -> tuple[Series, Series]:
    """Unsigned Differences
    Returns two Series, an unsigned positive and unsigned negative series based
    on the differences of the original series. The positive series are only the
    increases and the negative series are only the decreases.

    Default Example:
    series   = Series([3, 2, 2, 1, 1, 5, 6, 6, 7, 5, 3]) and returns
    postive  = Series([0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0])
    negative = Series([0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1])
    """
    asint = _bool_param(asint, False, "asint")
    amount = _pos_int(amount, 1, "amount")
    negative = series.diff(amount)
    negative.fillna(0, inplace=True)
    positive = negative.copy()

    positive[positive <= 0] = 0
    positive[positive > 0] = 1

    negative[negative >= 0] = 0
    negative[negative < 0] = 1

    if asint:
        positive = positive.astype(int)
        negative = negative.astype(int)

    return positive, negative


def verify_series(series: Series, min_length: float | None = None) -> Series | None:
    """If a Pandas Series and it meets the min_length of the indicator return it.

    Returns None for a Series shorter than *min_length* (an ordinary data
    condition) and for ``None`` (an optional argument that was not given).
    Indicators wrapped in :func:`nan_on_short_input` turn the short-input None
    into an all-NaN result for their callers.

    Anything else -- a list, a numpy array, a DataFrame -- is a caller error
    and raises TypeError, so the mistake surfaces where it was made rather than
    as a missing column or an unrelated error several frames later.
    """
    has_length = min_length is not None and isinstance(min_length, int)
    if series is not None and isinstance(series, Series):
        if has_length and series.size < min_length:
            logger.warning(f"[X] Series has {series.size} rows but indicator requires" f" at least {min_length}; the result is all NaN.")
            return None
        return series
    if series is not None:
        indicator = sys._getframe(1).f_code.co_name  # the indicator that called verify_series
        raise TypeError(
            f"{indicator}() expected a pandas Series but got {type(series).__name__}. "
            "Pass a Series, e.g. df['close'] rather than df['close'].values."
        )
    return None


def _sliding_weighted_ma(close: Series, length: int, weights: Any) -> Series:
    """Vectorised weighted MA via sliding_window_view.

    Args:
        close: The input series.
        length: Window length (must equal ``len(weights)``).
        weights: 1-D weight array whose orientation matches the window layout.

    Returns:
        A Series aligned with *close*, with ``NaN`` for the first
        ``length - 1`` positions.
    """
    arr = close.to_numpy(dtype=float)
    result = np.full(len(arr), np.nan)
    if length <= arr.shape[0]:
        windows = np.lib.stride_tricks.sliding_window_view(arr, length)
        result[length - 1 :] = windows @ weights
    return Series(result, index=close.index)


def _sliding_argextreme(series: Series, length: int, argfunc: Any, reverse: bool = False) -> Series:
    """Vectorised rolling ``argfunc`` (arg-position of a window extreme).

    Bit-identical to ``series.rolling(length).apply(argfunc, raw=True)`` — an
    integer arg-position over each window, with ``NaN`` for the first
    ``length - 1`` warm-up bars. With ``reverse=True`` the window is flipped
    first, matching ``argfunc(x[::-1])`` (periods since the most-recent extreme).

    Args:
        series: The input series.
        length: Rolling window length.
        argfunc: ``np.argmax`` or ``np.argmin``.
        reverse: Flip each window before applying *argfunc*.
    """
    arr = series.to_numpy(dtype=float)
    m = arr.shape[0]
    result = np.full(m, np.nan)
    if length <= m:
        windows = np.lib.stride_tricks.sliding_window_view(arr, length)
        if reverse:
            windows = windows[:, ::-1]
        result[length - 1 :] = argfunc(windows, axis=1)
    return Series(result, index=series.index)
