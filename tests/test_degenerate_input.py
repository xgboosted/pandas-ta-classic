"""Degenerate input: a window with no range, no movement and no variance.

``SPY_D`` contains 5241 bars and not one of them is flat -- no bar has
``high == low``, and no 14-bar window has a zero range. The whole degenerate
branch of every indicator is therefore unreached by the oracle suite, and
``test_oracle_talib.py`` could not have caught it anyway: its ``_compare``
intersects the two sides *after* ``dropna()``, so "native returns NaN where
TA-Lib returns a number" drops out of the comparison by construction.

What the degenerate case is
---------------------------
Every range-normalised indicator divides by a spread, a sum of absolute moves
or a standard deviation. On a flat window all three are exactly zero, and the
numerator usually is too, so the division is ``0 / 0``. 22 indicators returned
an all-``NaN`` column for it, and four more (``adx``, ``adxr``, ``qqe``) lost
their main value column while their other columns kept working.

The convention
--------------
TA-Lib writes a literal ``0.0`` whenever the denominator is exactly zero,
whatever ``0.0`` means on that indicator's scale -- ``WILLR`` 0.0 is the top of
its ``-100..0`` range, ``STOCH`` 0.0 the bottom of its ``0..100`` one. It is a
marker for "this window was degenerate", not a reading. This package follows
it, so that every indicator with a TA-Lib counterpart agrees with it bar for
bar, and the indicators without one answer the same way.

Note that this is *not* ``non_zero_range``, which substitutes an epsilon and
lets the value fall out of the formula. The two agree wherever the numerator is
also zero, and disagree where the formula has an affine term: with an epsilon
denominator ``willr`` reads ``-100``, with the zero convention ``0.0``.

Indicators left out
-------------------
``hilo`` and ``td_seq`` return ``NaN`` because nothing happened, not because a
division failed: the Gann activator has no state until the first breakout, and
a TD setup needs four bars of movement to start counting. Same for the sparse
signal columns of ``psar``, ``qqe`` and ``supertrend``, which mark the bars
where a signal fired and are ``NaN`` everywhere else by design.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

import pandas_ta_classic as ta
from tests.assertions import output_columns

try:
    import talib

    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False


# 300 rather than 250: vfi chains two 130-bar windows and produces nothing at
# all below 262 rows, on flat and on moving input alike, which would look like
# a degenerate-input failure here.
_N_ROWS = 300

# 0.5 rather than a price-like 100: it is a valid price and it also sits inside
# the domain of acos, asin and atanh, so the math category takes part in the
# registry sweep below instead of returning NaN for an unrelated reason.
_FLAT_VALUE = 0.5


def _index() -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=_N_ROWS, freq="D")


def _flat_frame() -> dict[str, pd.Series]:
    """Every bar identical: no range, no movement, no variance."""
    return _ohlcv(pd.Series(_FLAT_VALUE, index=_index()), 0.0)


def _moving_close() -> pd.Series:
    rng = np.random.default_rng(0)
    return pd.Series(100 + np.cumsum(rng.normal(0, 1, _N_ROWS)), index=_index())


def _ohlcv(close: pd.Series, spread: float) -> dict[str, pd.Series]:
    index = close.index
    return {
        "open_": close,
        "open": close,
        "high": close + spread,
        "low": close - spread,
        "close": close,
        "volume": pd.Series(1000.0, index=index),
        "benchmark": close,
        "series_a": close,
        "series_b": close,
        "source": close,
    }


def _block_frame(*, flat: bool = True) -> dict[str, pd.Series]:
    """A 40-bar flat block inside otherwise moving data.

    Windows are degenerate only while they sit entirely inside the block, so
    this catches a guard that fixes the fully flat case and still bleeds NaN
    into the windows that follow a flat run. ``flat=False`` builds the control:
    the same data without the block.
    """
    close = _moving_close()
    frame = _ohlcv(close, 1.0)
    if flat:
        for key in ("open_", "open", "high", "low", "close", "benchmark", "series_a", "series_b", "source"):
            frame[key] = frame[key].copy()
            frame[key].iloc[100:140] = 100.0
    return frame


def _no_range_frame(*, flat: bool = True) -> dict[str, pd.Series]:
    """Moving close, but ``high == low == close`` on every bar.

    Valid OHLC (a bar that traded at one price), and the spread dividers see a
    zero denominator on every single bar while the movement dividers do not.
    ``flat=False`` builds the control: the same close with a real spread.
    """
    return _ohlcv(_moving_close(), 0.0 if flat else 1.0)


_DEGENERATE_FRAMES = {"block": _block_frame, "no_range": _no_range_frame}


@pytest.fixture(scope="module")
def frames() -> dict[str, dict[str, pd.Series]]:
    built = {"flat": _flat_frame()}
    for label, build in _DEGENERATE_FRAMES.items():
        built[label] = build(flat=True)
        built[f"{label}:control"] = build(flat=False)
    return built


def _call(name: str, frame: dict[str, pd.Series], **kwargs):
    func = getattr(ta, name)
    series_kwargs = {p: frame[p] for p in inspect.signature(func).parameters if p in frame}
    return func(**series_kwargs, **_extra_kwargs(name, frame), **kwargs)


def _extra_kwargs(name: str, frame: dict[str, pd.Series]) -> dict:
    """Arguments with no sensible default for a blind sweep."""
    if name == "ma":
        return {"name": "sma"}
    if name in ("long_run", "short_run"):
        return {"fast": frame["close"], "slow": frame["close"]}
    if name == "mavp":
        return {"periods": pd.Series(10.0, index=frame["close"].index)}
    if name == "tsignals":
        return {"trend": pd.Series(1, index=frame["close"].index)}
    if name == "xsignals":
        return {"signal": frame["close"], "xa": 0.6, "xb": 0.4}
    return {}


# ---------------------------------------------------------------------------
# Indicators with a TA-Lib counterpart: the convention is checked against it.
# ---------------------------------------------------------------------------


# The TA-Lib function is called directly rather than through `talib=True`.
# Comparing the two paths of the same call would pass vacuously whenever the
# TA-Lib branch declines to run -- it requires every parameter it cannot
# express to be at its default (rule 6 in AGENTS.md), so a native-vs-native
# comparison is one changed default away. Calling the library keeps the test
# dynamic: if TA-Lib ever changes what it returns for a degenerate window,
# these fail rather than agreeing with a stale constant.
#
# stochrsi is absent on purpose: its TA-Lib branch folds the %K range and its
# smoothing into one period and is only taken at k=1, so the two are not the
# same function. It is pinned by value below instead.
def _talib_flat(name: str, frame: dict[str, pd.Series]):
    high, low, close, volume = frame["high"], frame["low"], frame["close"], frame["volume"]
    return {
        "adx": lambda: talib.ADX(high, low, close, 14),
        "adxr": lambda: talib.ADXR(high, low, close, 14),
        "atr": lambda: talib.ATR(high, low, close, 14),
        "cci": lambda: talib.CCI(high, low, close, 14),
        "cmo": lambda: talib.CMO(close, 14),
        "mfi": lambda: talib.MFI(high, low, close, volume, 14),
        "natr": lambda: talib.NATR(high, low, close, 14),
        "rsi": lambda: talib.RSI(close, 14),
        "uo": lambda: talib.ULTOSC(high, low, close, 7, 14, 28),
        "willr": lambda: talib.WILLR(high, low, close, 14),
    }[name]()


_ORACLE_CASES = ("adx", "adxr", "atr", "cci", "cmo", "mfi", "natr", "rsi", "uo", "willr")


@pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib is not installed")
@pytest.mark.parametrize("name", _ORACLE_CASES)
def test_flat_window_matches_talib(name: str, frames) -> None:
    """On a flat series the native path equals TA-Lib, bar for bar.

    Every post-warmup bar of the flat frame is degenerate, so this compares
    the convention and not the formula: ``cmo`` smooths differently from
    TA-Lib on real data, but on a flat series both are 0.

    The comparison runs over the bars where *TA-Lib* produced a number and
    requires a number there from the native path too. Intersecting the two
    non-NaN sets instead, which is what ``test_oracle_talib.py`` does, is
    exactly what hid this: it drops every bar where one side is NaN.
    """
    oracle = pd.Series(np.asarray(_talib_flat(name, frames["flat"]), dtype=float), index=frames["flat"]["close"].index)
    where = oracle.notna()
    assert where.any(), f"{name}: TA-Lib returned nothing to compare against"

    # The indicator's own column, whatever it is called; for adx and adxr the
    # frame also carries DMP/DMN, which TA-Lib exposes as separate functions.
    native_frame = output_columns(_call(name, frames["flat"], talib=False))
    key = next(c for c in native_frame if c.split("]", 1)[-1].upper().startswith(name.upper()))
    native = native_frame[key]

    missing = int((where & native.isna()).sum())
    assert not missing, f"{key}: NaN on {missing} bars where TA-Lib returned a number"
    difference = float((native[where] - oracle[where]).abs().max())
    assert difference < 1e-9, f"{key}: max abs diff {difference:.4e} against TA-Lib on a flat series"


# ---------------------------------------------------------------------------
# Indicators without a TA-Lib counterpart: the same convention, pinned here.
# ---------------------------------------------------------------------------

_CONVENTION_CASES = [
    "beta",
    "cdl_z",
    "chop",
    "correl",
    "cti",
    "cvi",
    "er",
    "inertia",
    "kurtosis",
    "qqe",
    "rvi",
    "skew",
    "smi",
    # stochrsi has a TA-Lib counterpart, but its branch folds the %K range and
    # its smoothing into one period and is only taken at k=1, so the two are
    # not the same function. Pinned by value here; the like-for-like oracle
    # comparison lives in test_oracle_talib.py.
    "stochrsi",
    "tsi",
    # TA-Lib's TRANGE agrees bar for bar, but the oracle sweep above looks the
    # native column up by the function name and this one is TRUERANGE_1.
    "true_range",
    "vfi",
    "vhf",
    "zscore",
]

# qqe's QQEl/QQEs mark the bars a long/short signal fired and are NaN elsewhere.
_SPARSE_COLUMN_PREFIXES = ("QQEl", "QQEs", "PSARl", "PSARs", "SUPERTl", "SUPERTs", "HILOl", "HILOs", "TD_SEQ")

# QQEd is a trend direction (+1 / -1), not a normalised value, so "the
# degenerate window reads 0.0" does not apply to it. It is still required to
# produce a value, by the sweeps below.
_CATEGORICAL_COLUMN_PREFIXES = ("QQEd",)


def _starts_with(column: str, prefixes: tuple[str, ...]) -> bool:
    return column.split("]", 1)[-1].startswith(prefixes)


@pytest.mark.parametrize("name", _CONVENTION_CASES)
def test_flat_window_reads_zero_without_an_oracle(name: str, frames) -> None:
    """A degenerate window reads 0.0, the same convention TA-Lib applies."""
    for column, values in output_columns(_call(name, frames["flat"])).items():
        if _starts_with(column, _SPARSE_COLUMN_PREFIXES):
            continue
        assert not values.isna().all(), f"{column}: all NaN on a flat series"
        if _starts_with(column, _CATEGORICAL_COLUMN_PREFIXES):
            continue
        finite = values.dropna()
        assert (finite == 0.0).all(), f"{column}: degenerate window reads {sorted(set(finite))[:5]}, expected 0.0"


# Each moment needs a minimum window: the skew divides by (n - 2), the excess
# kurtosis by (n - 2)(n - 3). Below that the formula is undefined, which is not
# the same as a degenerate window, so those bars read NaN and the bound raises.


# ---------------------------------------------------------------------------
# Registry sweep: a new indicator is covered the moment it is registered.
# ---------------------------------------------------------------------------

# NaN here means "nothing happened", not "the division failed". The Gann
# activator has no state until the first breakout and a TD setup needs four
# bars of movement, so neither can produce a value on a series that never
# moves. Their columns are all NaN together, which is why they are listed by
# name rather than by column.
_NO_SIGNAL_ON_FLAT_INPUT = {"hilo", "td_seq"}


def _indicator_names() -> list[str]:
    return sorted({n for names in ta.Category.values() for n in names if callable(getattr(ta, n, None)) and not inspect.isclass(getattr(ta, n))})


@pytest.mark.parametrize("name", [n for n in _indicator_names() if n not in _NO_SIGNAL_ON_FLAT_INPUT])
def test_no_column_is_all_nan_on_a_flat_series(name: str, frames) -> None:
    """A flat series is degenerate input, not missing data."""
    result = _call(name, frames["flat"])
    if result is None:
        pytest.skip(f"{name} returns None for this input")
    for column, values in output_columns(result).items():
        if _starts_with(column, _SPARSE_COLUMN_PREFIXES):
            continue
        assert not values.isna().all(), f"{column}: all NaN on a flat series"


@pytest.mark.parametrize("label", ["flat", *sorted(_DEGENERATE_FRAMES)])
@pytest.mark.parametrize("name", [n for n in _indicator_names() if n not in _NO_SIGNAL_ON_FLAT_INPUT])
def test_no_column_is_infinite_on_a_degenerate_window(name: str, label: str, frames) -> None:
    """The marker for a degenerate window is 0.0, never an infinity.

    "Not all NaN" does not cover this: a zero that reaches a logarithm, or the
    numerator of a ratio whose denominator is also zero, yields +-inf, which
    reads as a real and enormous value and passes every NaN check above. Both
    `chop` (log of HH - LL) and `vhf` (non_zero_range in the numerator) did.
    """
    result = _call(name, frames[label])
    if result is None:
        pytest.skip(f"{name} returns None for this input")
    for column, values in output_columns(result).items():
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        assert not np.isinf(numeric).any(), f"{column}: {int(np.isinf(numeric).sum())} infinite values on the {label} frame"


def _interior_nan(values: pd.Series) -> int | None:
    """NaN sitting after the column's first valid value; None if it has none."""
    first = values.first_valid_index()
    return None if first is None else int(values.loc[first:].isna().sum())


@pytest.mark.parametrize("label", sorted(_DEGENERATE_FRAMES))
@pytest.mark.parametrize("name", [n for n in _indicator_names() if n not in _NO_SIGNAL_ON_FLAT_INPUT])
def test_a_degenerate_window_does_not_bleed_nan(name: str, label: str, frames) -> None:
    """A flat run costs no bar beyond itself.

    Without a guard the NaN from a degenerate window spreads over the next
    ``length - 1`` windows, so a flat block of *b* bars costs
    ``max(0, b - (length - 1))`` extra NaN.

    The count is compared against a control frame built from the same data
    without the degenerate part, rather than required to be zero: ``dpo``
    centres its SMA and ``ichimoku`` shifts its Chikou span back, so both carry
    trailing NaN by design on any input, and only an *increase* is a bleed.
    """
    result = _call(name, frames[label])
    control = _call(name, frames[f"{label}:control"])
    if result is None or control is None:
        pytest.skip(f"{name} returns None for this input")

    control_columns = output_columns(control)
    for column, values in output_columns(result).items():
        if _starts_with(column, _SPARSE_COLUMN_PREFIXES) or column not in control_columns:
            continue
        expected = _interior_nan(control_columns[column])
        if expected is None:
            continue  # the indicator produces nothing on this frame either way
        actual = _interior_nan(values)
        assert actual is not None, f"{column}: all NaN on the {label} frame, {expected} interior NaN on the control"
        assert actual <= expected, f"{column}: {actual} NaN after the first valid bar on the {label} frame, {expected} on the control"
