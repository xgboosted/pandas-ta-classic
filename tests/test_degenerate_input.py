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

Note that this is *not* what ``non_zero_range`` did (removed in 0.9.0): it
substituted an epsilon and let the value fall out of the formula. The two agree
wherever the numerator is also zero, and disagree where the formula has an
affine term: with an epsilon denominator ``willr`` reads ``-100``, with the
zero convention ``0.0``. The properties at the end of this module keep the
epsilon from coming back.

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
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays

import pandas_ta_classic as ta
from tests.assertions import output_columns
from tests.test_lookahead import candidate_calls

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


def _inconsistent_block_frame(*, flat: bool = True) -> dict[str, pd.Series]:
    """A 40-bar block with ``high == low`` while ``open`` sits 0.2 below it.

    Malformed on purpose: valid OHLC cannot produce it, because ``high == low``
    forces ``open == close``. That is exactly why it is here -- on every other
    frame a zero denominator arrives with a zero numerator, which is where an
    epsilon and the 0.0 convention agree, so those frames cannot tell a guarded
    divider from an unguarded one. Nothing in the library validates the
    boundary, so such a bar is accepted and has to answer. ``flat=False``
    builds the control: the same data keeping its spread throughout.
    """
    close = _ramp_with_flat_close()
    high, low = close + 0.5, close - 0.5
    if flat:
        high.iloc[_SPREAD_BLOCK] = 100.0
        low.iloc[_SPREAD_BLOCK] = 100.0
    open_ = close - 0.2
    index = close.index
    return {
        "open_": open_,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": pd.Series(1000.0, index=index),
        "benchmark": close,
        "series_a": close,
        "series_b": close,
        "source": close,
    }


_DEGENERATE_FRAMES = {
    "block": _block_frame,
    "no_range": _no_range_frame,
    "inconsistent": _inconsistent_block_frame,
}


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
    # A flat series has no bar-to-bar direction, so no signed volume: bar 0 used
    # to count as a rising bar and left a decaying -9.6 instead.
    "kvo",
    "marketfi",
    "massi",
    "pdist",
    "qqe",
    "qstick",
    "rvgi",
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


# cdl_z(full=True) is anchored: its first bar has no deviation yet. hwc has no warm-up.
@pytest.mark.parametrize(("name", "kwargs", "prefix", "warm_up"), [("cdl_z", {"full": True}, "", 1), ("hwc", {"channel_eval": True}, "HWPCT", 0)])
def test_an_option_path_reads_zero_on_a_flat_series(name: str, kwargs: dict, prefix: str, warm_up: int, frames) -> None:
    """The option paths the default-only sweep missed follow the same 0.0 convention."""
    for column, values in output_columns(_call(name, frames["flat"], **kwargs)).items():
        if not _starts_with(column, (prefix,)):
            continue
        finite = values.dropna()
        assert len(finite) == len(values) - warm_up, f"{column}: {values.isna().sum()} NaN, expected {warm_up}"
        assert (finite == 0.0).all(), f"{column}: flat series reads {sorted(set(finite))[:5]}, expected 0.0"


@pytest.mark.parametrize("flat_value", [0.3, 100.1])
def test_a_flat_block_reads_zero_when_the_value_is_not_exact(flat_value: float) -> None:
    """A flat block at a value binary float cannot represent exactly leaves a
    ~1e-17 residue in the computed variance, so an exact ``== 0`` misses it and
    a degenerate window divides into a nonsense moment or band position. The
    block's windows must still read 0.0."""
    length = 10
    close = _moving_close()
    close.iloc[100:140] = flat_value
    frame = _ohlcv(close, 1.0)
    pos = 139  # the window ending here is entirely inside the flat block
    for name, kwargs, key in (
        ("zscore", {"length": length}, "ZS_10"),
        ("skew", {"length": length}, "SKEW_10"),
        ("kurtosis", {"length": length}, "KURT_10"),
        ("linreg", {"length": length, "r": True}, None),
        ("bbands", {"length": length}, "BBP_10_2.0"),
    ):
        result = _call(name, frame, **kwargs)
        values = result if isinstance(result, pd.Series) else result[key]
        assert values.iloc[pos] == 0.0, f"{name} flat {flat_value} reads {values.iloc[pos]}, expected 0.0"


# Each moment needs a minimum window: the skew divides by (n - 2), the excess
# kurtosis by (n - 2)(n - 3). Below that the formula is undefined, which is not
# the same as a degenerate window, so those bars read NaN and the bound raises.
_NARROW_WINDOW_CASES = [("kurtosis", 4), ("skew", 3)]


@pytest.mark.parametrize(("name", "minimum"), _NARROW_WINDOW_CASES)
def test_a_window_too_narrow_for_the_formula_is_not_degenerate(name: str, minimum: int) -> None:
    """Undefined is not degenerate, so it reads NaN rather than 0.0.

    ``kurtosis`` divides by ``(n - 2) * (n - 3) * m2**2``, which is zero for
    every window at ``length`` 2 or 3 however much the data moves: keying the
    guard on that divisor read 0.0 for all of them. Both are keyed on the
    variance instead, and ``length`` carries the bound. The sweeps above call
    each indicator at its default length, so neither would catch this.
    """
    close = _moving_close()
    func = getattr(ta, name)
    with pytest.raises(ValueError, match=rf"{name}\(\) length must be an integer > {minimum - 1}"):
        func(close, length=minimum - 1)

    # min_periods may still let a narrower window through; those bars have no
    # moment to report and must not read 0.0 or +-inf either.
    narrow = func(close, length=10, min_periods=2)
    head, tail = narrow.iloc[: minimum - 1], narrow.iloc[minimum - 1 :]
    assert head.isna().all(), f"{name}: a window under {minimum} bars reads {head.tolist()}, expected NaN"
    assert np.isfinite(tail).all(), f"{name}: {tail[~np.isfinite(tail)].tolist()} where the formula is defined"


@pytest.mark.parametrize(("name", "minimum"), _NARROW_WINDOW_CASES)
def test_min_periods_is_bounded_and_the_tail_matches_pandas(name: str, minimum: int) -> None:
    """``min_periods`` is bounded to 1..length, and the tail matches pandas.

    ``0`` read the loop start as ``-1`` and made the last bar NaN; ``> length``
    was silently ignored. Both now raise. The partial-window tail must agree
    with ``close.rolling(length, min_periods=...).kurt()``/``.skew()``.
    """
    close = _moving_close()
    func = getattr(ta, name)
    for bad in (0, 11):
        with pytest.raises(ValueError, match=rf"{name}\(\) min_periods must be an integer >= 1 and < 11"):
            func(close, length=10, min_periods=bad)
    for mp in (1, 2, 5, 10):
        native = func(close, length=10, min_periods=mp)
        pandas_ref = close.rolling(10, min_periods=mp).kurt() if name == "kurtosis" else close.rolling(10, min_periods=mp).skew()
        aligned = native.dropna()
        assert np.allclose(
            aligned.to_numpy(), pandas_ref.loc[aligned.index].to_numpy(), equal_nan=True
        ), f"{name} min_periods={mp} disagrees with pandas rolling"


# ---------------------------------------------------------------------------
# Registry sweep: a new indicator is covered the moment it is registered.
# ---------------------------------------------------------------------------

# NaN here means "nothing happened", not "the division failed". The Gann
# activator has no state until the first breakout and a TD setup needs four
# bars of movement, so neither can produce a value on a series that never
# moves. Their columns are all NaN together, which is why they are listed by
# name rather than by column.
_NO_SIGNAL_ON_FLAT_INPUT = {"hilo", "td_seq"}

# A live numerator over a zero denominator is a real division by zero, not a
# degenerate window: it reads +-inf (as it did before the degenerate-window
# sweep), because the mask keys on *both* operands being zero. ``brar`` and
# ``chop`` reach it at the boundary of a flat block, where the highs outlive
# the lows on the block and inconsistent frames.
_REAL_DIVISION_BY_ZERO = {"brar", "chop"}


def _indicator_names() -> list[str]:
    return sorted({n for names in ta.Category.values() for n in names if callable(getattr(ta, n, None)) and not inspect.isclass(getattr(ta, n))})


# The default call plus each boolean keyword flipped on its own: a guard added on
# the default path misses an option path with its own division -- the default-only
# sweep let ``cdl_z(full=True)`` and ``hwc(channel_eval=True)`` stay all NaN.
_FLAT_SERIES_CALLS = [(n, kwargs) for n in _indicator_names() if n not in _NO_SIGNAL_ON_FLAT_INPUT for kwargs in candidate_calls(n)]


@pytest.mark.parametrize(("name", "kwargs"), _FLAT_SERIES_CALLS, ids=[f"{n}-{k}" if k else n for n, k in _FLAT_SERIES_CALLS])
def test_no_column_is_all_nan_on_a_flat_series(name: str, kwargs: dict, frames) -> None:
    """A flat series is degenerate input, not missing data."""
    result = _call(name, frames["flat"], **kwargs)
    if result is None:
        pytest.skip(f"{name} returns None for this input")
    for column, values in output_columns(result).items():
        if _starts_with(column, _SPARSE_COLUMN_PREFIXES):
            continue
        assert not values.isna().all(), f"{column}: all NaN on a flat series"


_CHANNEL_CASES = [("kc", {"tr": True}), ("kc", {"tr": False}), ("accbands", {})]


@pytest.mark.parametrize(("name", "kwargs"), _CHANNEL_CASES, ids=lambda v: str(v))
def test_a_flat_series_gives_a_channel_no_width(name: str, kwargs: dict, frames) -> None:
    """A band is a reported range, so a flat bar widens it by exactly 0.

    These channels sit at the price, not at 0.0, so the convention sweep does
    not cover them. ``kc`` has to agree with itself across both range sources:
    ``tr=True`` goes through ``true_range``, ``tr=False`` took ``non_zero_range``
    directly, and ``accbands`` scales the range by ``high + low``. All three
    reported an epsilon-wide channel where there was no range at all.
    """
    channel = _call(name, frames["flat"], **kwargs).dropna()
    width = (channel.iloc[:, 2] - channel.iloc[:, 0]).abs()
    assert not width.empty, f"{name} produced nothing to check"
    assert (width == 0).all(), f"{name}{kwargs} widens a flat channel by {width.max():.3e}"


@pytest.mark.parametrize("name", ["emv", "eom", "massi"])
def test_a_bar_with_no_range_reads_zero_while_the_close_still_moves(name: str, frames) -> None:
    """The no_range frame is the one that reaches these three.

    ``emv`` and ``eom`` carry the range as a numerator factor, so on the flat
    frame their distance term is zero as well and they answer 0.0 whether the
    range is exact or an epsilon. Only a moving close with ``high == low`` on
    every bar separates the two, and there the epsilon surfaced: 6.9e-18 for
    ``emv``, 1.8e-14 for ``eom``. ``massi`` divides two EMAs of the range, which
    an epsilon made exactly 1, summing to ``slow`` -- 25, an ordinary reading.
    """
    for column, values in output_columns(_call(name, frames["no_range"])).items():
        finite = values.dropna()
        assert not finite.empty, f"{column}: nothing to check"
        assert (finite == 0.0).all(), f"{column}: reads {sorted(set(finite))[:5]} with no range, expected 0.0"


def test_a_bar_that_traded_at_one_price_on_no_volume_reads_zero() -> None:
    """``marketfi`` divides a range by volume, and both can be zero at once.

    The frames above all carry volume, so only this reaches the 0/0: a halted
    session prints one price and no trades. It reads 0.0 like any degenerate
    window, where the epsilon had made it ``eps / 0 == inf``. A real range on no
    volume is a genuine division by zero and still reports one.
    """
    index = pd.date_range("2020-01-01", periods=40, freq="D")
    price = pd.Series(_FLAT_VALUE, index=index)
    halted = ta.marketfi(price, price, pd.Series(0.0, index=index)).dropna()
    assert not halted.empty, "marketfi produced nothing to check"
    assert (halted == 0.0).all(), f"marketfi reads {sorted(set(halted))[:5]} for a halted bar, expected 0.0"

    ranged = ta.marketfi(price + 1.0, price, pd.Series(0.0, index=index)).dropna()
    assert np.isinf(ranged).all(), "a real range on no volume is still a division by zero"


def test_a_flat_series_gives_bollinger_no_width_and_no_position(frames) -> None:
    """``BBB`` is a width, so 0.0 is its value; ``BBP``'s 0/0 takes the marker.

    The three bands themselves sit at the price, so the convention sweep does
    not reach this one. ``non_zero_range`` had left ``BBB`` at ``100 * eps / mid``
    and made ``BBP`` ``eps / eps == 1.0``, "close at the upper band", which is no
    truer on a constant window than 0.0 at the lower one.
    """
    for key, values in output_columns(_call("bbands", frames["flat"])).items():
        if not _starts_with(key, ("BBB", "BBP")):
            continue
        finite = values.dropna()
        assert not finite.empty, f"{key}: nothing to check"
        assert (finite == 0.0).all(), f"{key}: reads {sorted(set(finite))[:5]} on a flat series, expected 0.0"


# ht_phasor is deliberately left alone. It carries no epsilon: its Hilbert FIR
# evaluates `0.0962*x[i] + 0.5769*x[i-2] - 0.5769*x[i-4] - 0.0962*x[i-6]` left to
# right, so a constant input leaves 6e-17 of rounding where TA-Lib reaches
# exactly 0. Grouping the antisymmetric pairs would cancel it, but measured over
# SPY_D that moves ht_phasor, ht_dcperiod, ht_dcphase and ht_sine on nearly every
# bar by up to 7.6e-13, and it does not improve agreement with TA-Lib (the
# deviation there is already 2e-11, four orders above the residue). Not worth the
# churn in a file with known cross-OS ULP drift.


def test_a_window_with_range_but_no_movement_is_not_marked() -> None:
    """0.0 is the degenerate marker; it may not stand in for a real 1/0.

    ``vhf`` divides a window's range by the rolling sum of ``|close.diff(drift)|``.
    At the default ``drift=1`` no movement implies no range, so the two zeros
    arrive together. At ``drift >= 2`` an exactly periodic close has every diff
    zero while the range remains, and marking that 0.0 would report "not
    trending" for a window that is, in the limit, infinitely trending.
    """
    close = pd.Series([1.0, 2.0] * 15, index=pd.date_range("2020-01-01", periods=30, freq="D"))
    result = ta.vhf(close, length=10, drift=2).dropna()
    assert result.any(), "vhf produced nothing to check"
    assert (result != 0.0).all(), f"vhf marked a window that still has range: {sorted(set(result))[:5]}"


@pytest.mark.parametrize("label", ["flat", *sorted(_DEGENERATE_FRAMES)])
@pytest.mark.parametrize("name", [n for n in _indicator_names() if n not in _NO_SIGNAL_ON_FLAT_INPUT and n not in _REAL_DIVISION_BY_ZERO])
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


# An unguarded divider multiplies by 1/epsilon = 4.5e15. A legitimate blow-up
# stays far below that: the largest here is `cvi`, a rate of change whose EMA
# base decays toward zero without reaching it as a flat block ends, and it
# reaches 2.6e5 against a control of 0. Nine orders of magnitude separate the
# two, so the bound needs no per-indicator exception and catching an epsilon
# never depends on tuning it.
_MAGNITUDE_HEADROOM = 1e9


def _finite_max(values: pd.Series) -> float | None:
    """Largest finite magnitude in a column; None when it has none."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return None if not finite.size else float(np.abs(finite).max())


@pytest.mark.parametrize("label", sorted(_DEGENERATE_FRAMES))
@pytest.mark.parametrize("name", [n for n in _indicator_names() if n not in _NO_SIGNAL_ON_FLAT_INPUT])
def test_a_degenerate_window_does_not_inflate_the_output(name: str, label: str, frames) -> None:
    """A degenerate window may not answer beyond the scale of the same data.

    This is the property the sweeps above cannot see. Dividing by the epsilon
    the removed ``non_zero_range`` substituted multiplies the quotient by 4.5e15, which is
    finite, is not NaN, and bleeds no NaN -- ``ad`` reached 3.6e23 that way and
    ``cumsum`` carried it to the end of the series, while every other check here
    passed. An epsilon *residue* hides just as well at the other end: 2.2e-16 is
    a perfectly ordinary-looking number too.

    The ceiling is the control frame's own largest value, so no indicator needs
    a constant of its own and a newly registered one is covered the moment it
    appears in the registry.
    """
    result = _call(name, frames[label])
    control = _call(name, frames[f"{label}:control"])
    if result is None or control is None:
        pytest.skip(f"{name} returns None for this input")

    control_columns = output_columns(control)
    for column, values in output_columns(result).items():
        if column not in control_columns:
            continue
        scale = _finite_max(control_columns[column])
        reached = _finite_max(values)
        if scale is None or reached is None:
            continue  # the NaN sweeps above own the "nothing came back" case
        # max(..., _FLAT_VALUE) keeps a control that is legitimately ~0 from
        # making the ceiling zero as well.
        allowed = _MAGNITUDE_HEADROOM * max(scale, _FLAT_VALUE)
        assert reached <= allowed, f"{column}: reaches {reached:.3e} on the {label} frame, {scale:.3e} on the control"


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


# ---------------------------------------------------------------------------
# A zero denominator with a non-zero numerator: the epsilon explodes.
# ---------------------------------------------------------------------------

# The frames above make the numerator zero wherever the denominator is, which
# is where an epsilon denominator and the 0.0 convention agree. Six indicators
# reach a zero denominator with a *non-zero* numerator, and there the epsilon
# does not degrade -- it multiplies by 4.5e15. Measured on the frames below,
# before the guards: ADo 3.6e19, ADOSC 3.2e18, CMF 9.0e14, BR 8.2e16,
# RVGI 9.0e14, BBP 1.5e14 (mamode="ema") and 6.7e14 ("rma").
#
# Only bbands is reachable with consistent OHLC. The other five need a bar that
# violates low <= open, close <= high, which no real feed should produce -- but
# accepting it and answering 1e19 is worse than accepting it and answering 0.0,
# and the library validates nothing at that boundary today.
_SPREAD_ROWS = 200
_SPREAD_BLOCK = slice(100, 140)


def _ramp_with_flat_close() -> pd.Series:
    index = pd.date_range("2020-01-01", periods=_SPREAD_ROWS, freq="D")
    close = pd.Series(np.linspace(90.0, 110.0, _SPREAD_ROWS), index=index)
    close.iloc[_SPREAD_BLOCK] = 100.0
    return close


def _inconsistent_frame() -> dict[str, pd.Series]:
    """high == low == close on the block, but open is 0.2 below it.

    Malformed: high == low forces open == close in valid OHLC. The point is
    that the answer stays bounded anyway.
    """
    close = _ramp_with_flat_close()
    high, low = close + 0.5, close - 0.5
    high.iloc[_SPREAD_BLOCK] = 100.0
    low.iloc[_SPREAD_BLOCK] = 100.0
    return {
        "open_": close - 0.2,
        "high": high,
        "low": low,
        "close": close,
        "volume": pd.Series(1000.0, index=close.index),
    }


def _flat_block_frame() -> dict[str, pd.Series]:
    """A fully flat block, OHLC-consistent on every bar including the edges.

    ``open`` is the previous close, so the bar entering the block still spans
    the jump into it and keeps its own spread.
    """
    close = _ramp_with_flat_close()
    open_ = close.shift(1)
    open_.iloc[0] = close.iloc[0]
    spread = pd.Series(0.5, index=close.index)
    spread.iloc[_SPREAD_BLOCK.start + 1 : _SPREAD_BLOCK.stop] = 0.0
    return {
        "open_": open_,
        "high": np.maximum(open_, close) + spread,
        "low": np.minimum(open_, close) - spread,
        "close": close,
        "volume": pd.Series(1000.0, index=close.index),
    }


def _assert_ohlc_consistent(frame: dict[str, pd.Series]) -> None:
    both = pd.concat([frame["open_"], frame["close"]], axis=1)
    bad = int((~((frame["low"] <= both.min(axis=1)) & (both.max(axis=1) <= frame["high"]))).sum())
    assert not bad, f"{bad} bars violate low <= open, close <= high"


def test_accumulation_stays_bounded_on_an_inconsistent_bar() -> None:
    """ad(open_=) and its dependents read 0.0 rather than 4.5e15 per bar.

    ``ad`` with ``open_`` normalises ``close - open_`` by the high-low range.
    On a bar with no range the numerator is not zero too, so an epsilon
    denominator gave ``0.2 / 2.2e-16`` -- and ``cumsum`` then froze it into
    every later bar, so a single malformed bar cost the whole series.
    """
    frame = _inconsistent_frame()
    ado = ta.ad(frame["high"], frame["low"], frame["close"], frame["volume"], open_=frame["open_"])

    # Each ordinary bar contributes (close - open_) * volume / (high - low)
    # = 0.2 * 1000 / 1.0 = 200; each degenerate bar contributes 0.0.
    assert ado.iloc[99] == pytest.approx(100 * 200.0)
    block = ado.iloc[_SPREAD_BLOCK]
    assert block.to_numpy() == pytest.approx(ado.iloc[99]), "a bar with no range moved the accumulation"
    assert ado.iloc[-1] == pytest.approx((_SPREAD_ROWS - 40) * 200.0)

    # adosc is built on ad, so it inherits the guard rather than repeating it.
    adosc = ta.adosc(frame["high"], frame["low"], frame["close"], frame["volume"], open_=frame["open_"])
    assert np.isfinite(adosc.dropna()).all()
    assert float(adosc.abs().max()) < 1e6, "ADOSC still carries the exploded accumulation"

    cmf = ta.cmf(frame["high"], frame["low"], frame["close"], frame["volume"], open_=frame["open_"])
    assert float(cmf.abs().max()) <= 1.0, "CMF is a fraction of volume and cannot exceed 1"


@pytest.mark.parametrize("name", ["rvgi"])
def test_a_rolling_sum_of_ranges_reads_zero_when_it_is_empty(name: str) -> None:
    """An epsilon survives a rolling sum as dust, not as a zero.

    ``rvgi`` divides by a rolling sum of bar ranges. With an epsilon in each
    term a window with no range at all sums to ~5e-15 rather than to zero, so
    the guard never fires and the quotient reaches 1e16. Its denominator is an
    exact difference for that reason. (``brar`` used to be here too; its
    numerator is exact now as well, and the real x/0 it reaches at a flat
    block's boundary is covered by ``test_brar_reads_inf_for_a_live_numerator``.)

    The frame has to be the inconsistent one. On a *consistent* flat block both
    sides of each ratio are epsilon dust and the quotient comes out near 1, so
    the fully flat frame passes this whether the guard is there or not.
    """
    frame = _inconsistent_frame()

    for column, values in output_columns(_call(name, frame)).items():
        finite = values.dropna()
        assert np.isfinite(finite).all(), f"{column}: {finite[~np.isfinite(finite)].tolist()}"
        assert float(finite.abs().max()) < 1e6, f"{column}: reaches {finite.abs().max():.3e} on a window with no range"


def test_brar_reads_inf_for_a_live_numerator_over_a_zero_denominator() -> None:
    """A gap-up bar leaves a live numerator over a zero denominator.

    ``open == low`` and ``high > open`` put a positive numerator over a zero
    exact denominator, which is a real division by zero and reads ``+inf`` --
    not the ``0.0`` a divisor-only mask would report, and not the epsilon dust
    the old ``non_zero_range`` numerator left (which read ``9.0e17``).
    """
    i = np.arange(40)
    open_ = pd.Series(100.0 + 3 * i)
    high = open_ + 2
    low = open_  # open == low, so every low is above the previous close
    close = open_ + 1
    result = ta.brar(open_, high, low, close, length=5)
    assert result["BR_5"].iloc[-1] == np.inf
    assert result["AR_5"].iloc[-1] == np.inf


@pytest.mark.parametrize("mamode", ["sma", "ema", "rma", "wma"])
def test_bbands_reads_zero_for_a_band_with_no_width(mamode: str) -> None:
    """A zero-width band reads 0.0 for both BBB and BBP, on every mamode.

    This one needs no malformed bar: a flat close run of ``length`` bars in
    otherwise moving, consistent OHLC gives a standard deviation of exactly
    zero, so ``upper == lower``. The epsilon width read BBB 2.2e-16 and, for the
    mamodes whose mid is exact, BBP = eps/eps = 1.0 -- a band with no top
    reporting the price at its top. ``ema`` and ``rma`` only converge on the
    constant, leaving a non-zero numerator, and BBP reached 1e14.
    """
    frame = _flat_block_frame()
    _assert_ohlc_consistent(frame)
    length = 5
    result = ta.bbands(frame["close"], length=length, mamode=mamode)

    # The window is degenerate once it sits entirely inside the flat block.
    degenerate = slice(_SPREAD_BLOCK.start + length - 1, _SPREAD_BLOCK.stop)
    for column in (f"BBB_{length}_2.0", f"BBP_{length}_2.0"):
        values = result[column].iloc[degenerate]
        assert (values == 0.0).all(), f"{column}: degenerate window reads {sorted(set(values))[:5]}, expected 0.0"

    assert float(result[f"BBP_{length}_2.0"].dropna().abs().max()) < 1e3, "BBP still explodes somewhere on this frame"


# ---------------------------------------------------------------------------
# Properties: a flat bar never leaves an epsilon in the output.
# ---------------------------------------------------------------------------

# ``non_zero_range`` (removed in 0.9.0) replaced a zero difference with
# ``sys.float_info.epsilon``. Each property below covers one way that epsilon
# reached the output, on generated frames with flat bars, bodiless bars, a
# fully flat run and, optionally, int64 prices. Prices sit on a 0.25 tick, so
# no legitimate value of the checked columns is epsilon-scale: anything with
# ``0 < |x| < 1e-12`` is the artifact.
_TICK = 0.25
_EPSILON_SCALE = 1e-12
_PROPERTY_LENGTH = 10


@st.composite
def _tick_frame(draw, *, consistent: bool = True) -> dict[str, pd.Series]:
    """OHLCV on a 0.25 tick with flat bars, bodiless bars and one flat run.

    ``consistent=False`` keeps ``high == low`` on the flat run but moves the
    close off it, the malformed bar on which a zero range meets a live
    numerator.
    """
    n = draw(st.integers(min_value=40, max_value=160))
    ticks = lambda lo, hi: draw(arrays(np.int64, n, elements=st.integers(lo, hi)))
    close = 400 + np.cumsum(ticks(-4, 4))
    high = close + ticks(0, 3) * draw(arrays(np.int64, n, elements=st.integers(0, 1)))
    low = close - ticks(0, 3) * draw(arrays(np.int64, n, elements=st.integers(0, 1)))
    open_ = low + ticks(0, 6) % (high - low + 1)
    bodiless = draw(arrays(bool, n))
    open_[bodiless] = close[bodiless]

    start = draw(st.integers(min_value=0, max_value=n - 2 * _PROPERTY_LENGTH))
    stop = draw(st.integers(min_value=start + 2 * _PROPERTY_LENGTH, max_value=n))
    price = close[start]
    open_[start:stop] = high[start:stop] = low[start:stop] = close[start:stop] = price
    if not consistent:
        close[start:stop] = price + draw(st.integers(min_value=1, max_value=4))

    volume = draw(arrays(np.int64, n, elements=st.integers(1, 10_000)))
    if draw(st.booleans()):  # int64 prices: the epsilon was once downcast to 0 there
        cast = lambda x: pd.Series(x, dtype="int64")
    else:
        cast = lambda x: pd.Series(x * _TICK, dtype="float64")
    return {"open_": cast(open_), "high": cast(high), "low": cast(low), "close": cast(close), "volume": pd.Series(volume, dtype="float64")}


def _epsilon_scale(values: pd.Series) -> pd.Series:
    return (values != 0) & (values.abs() < _EPSILON_SCALE)


@settings(max_examples=200, deadline=None)
@given(_tick_frame())
def test_a_bar_with_no_range_or_no_body_contributes_exactly_nothing(frame: dict[str, pd.Series]) -> None:
    """Per-bar ratios over the range: ``bop`` and the ``ad`` accumulation.

    ``bop`` read ``eps / range`` on a bodiless bar and ``ad(open_=)`` added
    ``eps * volume / range`` per such bar, where both are exactly 0.
    """
    o, h, l, c, v = frame["open_"], frame["high"], frame["low"], frame["close"], frame["volume"]
    nothing = (h == l) | (c == o)

    bop = ta.bop(o, h, l, c)
    assert (bop[nothing] == 0.0).all(), f"bop on a bar with no range or body: {sorted(set(bop[nothing]))[:5]}"
    assert not _epsilon_scale(bop).any()

    for ad in (ta.ad(h, l, c, v, open_=o), ta.ad(h, l, c, v)):
        step = ad.diff()
        step.iloc[0] = ad.iloc[0]
        moved = step[(h == l) | ((c == o) & (ad.name == "ADo"))]
        assert (moved == 0.0).all(), f"{ad.name} moved by {sorted(set(moved))[:5]} on a bar that adds nothing"


@settings(max_examples=200, deadline=None)
@given(_tick_frame(consistent=False))
def test_a_window_with_no_range_reads_exactly_zero(frame: dict[str, pd.Series]) -> None:
    """Rolling-range dividers: the 0.0 marker, not ``x / eps``.

    The frame is the malformed one, so the close can sit off a zero range:
    an epsilon divisor read 4.5e15 times the gap there, the mask reads 0.0.
    """
    h, l, c = frame["high"], frame["low"], frame["close"]
    no_range = (h.rolling(_PROPERTY_LENGTH).max() - l.rolling(_PROPERTY_LENGTH).min()) == 0
    for values in (
        ta.stochf(h, l, c, fastk=_PROPERTY_LENGTH).iloc[:, 0],
        ta.willr(h, l, c, length=_PROPERTY_LENGTH),
    ):
        assert (values[no_range] == 0.0).all(), f"{values.name} on a window with no range: {sorted(set(values[no_range]))[:5]}"
        assert not _epsilon_scale(values).any(), f"{values.name} carries an epsilon"


@settings(max_examples=200, deadline=None)
@given(_tick_frame())
def test_a_window_with_no_movement_closes_on_the_price_at_the_fast_rate(frame: dict[str, pd.Series]) -> None:
    """``kama``'s efficiency ratio is 1.0 on a window that did not move.

    That is TA-Lib's KAMA. An epsilon per diff made it ``eps / (length * eps)``,
    ``1 / length``, and the average crawled towards a flat price at nearly the
    slow rate instead: 0.085 off TA-Lib on a 40-bar flat block.
    """
    close = frame["close"].astype(float)
    kama = ta.kama(close, length=_PROPERTY_LENGTH)
    fast = (2 / 3) ** 2  # sc at er == 1 for the default fast=2
    still = close.diff().abs().rolling(_PROPERTY_LENGTH).sum() == 0
    for i in np.flatnonzero(still.to_numpy()):
        if i <= _PROPERTY_LENGTH - 1:
            continue
        expected = fast * close.iloc[i] + (1 - fast) * kama.iloc[i - 1]
        assert kama.iloc[i] == pytest.approx(expected, rel=1e-12), f"bar {i}: {kama.iloc[i]!r}, expected {expected!r}"


@settings(max_examples=200, deadline=None)
@given(_tick_frame())
def test_a_bar_with_no_body_is_always_a_doji(frame: dict[str, pd.Series]) -> None:
    """``body <= 0.01 * factor * average range`` holds for a zero body.

    The epsilon body failed it where the average range was 0 too -- inside a
    flat run -- so a flat bar was not a doji there. TA-Lib's CDLDOJI says it is.
    """
    o, h, l, c = frame["open_"], frame["high"], frame["low"], frame["close"]
    doji = ta.cdl_doji(o, h, l, c, length=_PROPERTY_LENGTH)
    bodiless = (c == o).to_numpy(copy=True)
    bodiless[: _PROPERTY_LENGTH + 1] = False  # the average range is warming up
    assert (doji[bodiless] == 100).all(), f"bodiless bars read {sorted(set(doji[bodiless]))}"


@pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
@settings(max_examples=100, deadline=None)
@given(_tick_frame())
def test_former_epsilon_readers_match_talib_on_flat_bars(frame: dict[str, pd.Series]) -> None:
    """Every module that used ``non_zero_range`` and has a TA-Lib twin agrees with it."""
    o, h, l, c = (frame[k].astype(float) for k in ("open_", "high", "low", "close"))
    n = _PROPERTY_LENGTH
    pairs = {
        "cdl_doji": (ta.cdl_doji(o, h, l, c, length=n), talib.CDLDOJI(o, h, l, c)),
        "bop": (ta.bop(o, h, l, c), talib.BOP(o, h, l, c)),
        "kama": (ta.kama(c, length=n), talib.KAMA(c, n)),
        "stochf": (ta.stochf(h, l, c, fastk=n).iloc[:, 0], talib.STOCHF(h, l, c, n, 3, 0)[0]),
        "dx": (ta.dx(h, l, c, length=n), talib.DX(h, l, c, n)),
    }
    for name, (native, oracle) in pairs.items():
        native, oracle = np.asarray(native, dtype=float), np.asarray(oracle, dtype=float)
        both = np.isfinite(native) & np.isfinite(oracle)
        np.testing.assert_allclose(native[both], oracle[both], rtol=1e-9, atol=1e-9, err_msg=name)


# Rounding residue, not a substituted epsilon. `ht_phasor`: the Hilbert FIR
# leaves 6e-17 on a constant input (see the Hilbert FIR note above).
# `cpr`: CPR_WIDTH is |TC - BC|, two pivot sums that round apart by 2.8e-14 on
# ordinary moving bars, flat or not. `stochrsi`: the %K/%D moving averages
# leave 1e-14 as the window enters a flat block, the usual rolling-sum residue.
_RESIDUE_NOT_EPSILON = {"cpr", "ht_phasor", "stochrsi"}


@pytest.mark.parametrize("label", ["flat", "block", "no_range"])
@pytest.mark.parametrize("name", [n for n in _indicator_names() if n not in _RESIDUE_NOT_EPSILON])
def test_no_column_carries_an_epsilon_scale_value(name: str, label: str, frames) -> None:
    """Registry-wide: a degenerate window answers 0.0 or a real value, never ~1e-16.

    The inflation sweep above catches an epsilon *divisor* with a live
    numerator; this catches the epsilon that reaches the output as itself, as
    ``bop``, ``ad``, ``adosc`` and ``cmf`` did through ``non_zero_range``
    before 0.9.0 (bodiless bars inside the block frame). The frames are
    consistent OHLC, where an exact formula leaves either 0.0 or a value on
    the data's own scale. The inconsistent frame is left out: on it, EMA-based
    columns legitimately converge on a constant and pass through 1e-13.
    """
    result = _call(name, frames[label])
    if result is None:
        pytest.skip(f"{name} returns None for this input")
    for column, values in output_columns(result).items():
        numeric = pd.to_numeric(values, errors="coerce")
        tiny = numeric[_epsilon_scale(numeric)]
        assert tiny.empty, f"{column}: {len(tiny)} epsilon-scale values on the {label} frame, e.g. {tiny.iloc[0]!r} at {tiny.index[0]}"
