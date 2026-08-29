"""Guards against future-data leakage (lookahead bias).

An indicator is *causal* when the value it reports at bar ``t`` depends only on
bars ``<= t``.  A causal indicator must therefore return identical values for
the first ``K`` bars whether it is evaluated over ``K`` bars or over the whole
series.  Any difference means a bar after ``K`` changed an earlier row, so a
backtest run over the full history would use information that did not exist at
the time the signal is claimed to have been produced.

Covered:
  1. Issue #149 — ``non_zero_range`` added its epsilon to *every* row as soon as
     any row in the batch had a zero range, so appending one flat bar shifted
     the whole history.  27 indicators inherited the leak.
  2. ``mavp`` — the default ``periods`` was ``linspace(min, max, len(close))``,
     so every bar's window length depended on the total number of bars.
  3. ``cdl_z(full=True)`` — a whole-series window plus ``bfill()`` copied the
     final bar's Z Score onto every earlier row.
  4. One parametrised case per registered indicator, at its defaults and at
     each boolean keyword flipped one at a time.

A leak is only visible if the input contains whatever the leaking branch tests
for -- issue #149 was invisible on the SPY sample data because that data has no
flat bar.  ``causal_test_data`` therefore plants a set of pathological bars,
all of them *after* the last cut point, so a causal indicator cannot see them
in any prefix.  Add to that set whenever a new batch-global branch appears.

Indicators are discovered through ``ta.Category``; nothing here enumerates
them.  The handful of calls that look forward on purpose are marked
``xfail(strict=True)`` through ``LOOKAHEAD_RULES``, so the exemptions cannot go
stale in either direction: an unmarked call that leaks fails, and a marked call
that has become causal fails as XPASS until its rule is narrowed or dropped.

Run:
    pytest tests/test_lookahead.py
"""

import inspect
import typing
from pathlib import Path
from sys import float_info as sflt

import numpy as np
import pandas as pd
import pytest

import pandas_ta_classic as ta
from pandas_ta_classic.utils import non_zero_range

# Number of bars in the synthetic frame, and the prefix lengths compared
# against it.  Two cuts so both a short and a long history are exercised.
BARS = 400
CUTS = (100, 250)
CUT = CUTS[-1]

# Boolean parameters the keyword sweep leaves alone.
SKIP_BOOL_KWARGS = {
    # selects a different implementation altogether, and whether TA-Lib is
    # installed varies by environment
    "talib",
    # df.ta.ichimoku() hardcodes as_dataframe=True and forwards **kwargs, so
    # passing it through the accessor raises TypeError on the duplicate
    "as_dataframe",
}

# Why a given call is expected to report values that depend on later bars.
# The predicate reads the keywords of the call, so a keyword that switches the
# forward-looking branch off (dpo(centered=False), ichimoku(include_chikou=
# False), cpr's default virgin_cpr=False) is *not* excused and is swept like
# any other call.
LOOKAHEAD_RULES = {
    "cpr": (
        lambda kwargs: kwargs.get("virgin_cpr", False),
        "a virgin CPR is defined by the next `virgin_lookforward` bars",
    ),
    "dpo": (
        lambda kwargs: kwargs.get("centered", True),
        "centered=True shifts the result back by int(0.5 * length) + 1",
    ),
    "ichimoku": (
        lambda kwargs: kwargs.get("include_chikou", True),
        "the Chikou span is close.shift(-kijun)",
    ),
    "tos_stdevall": (
        lambda _kwargs: True,
        "one linear regression is fitted over the entire series",
    ),
    "vp": (
        lambda _kwargs: True,
        "the whole series is aggregated into price bins; the result is not a time series",
    ),
}


def causal_test_data(bars: int = BARS) -> pd.DataFrame:
    """OHLCV frame whose pathological bars all sit after the last cut point.

    Each planted feature is the trigger for a branch that has historically been
    written batch-globally:

    * flat bars (``high == low``) -- the epsilon substitution in ``non_zero_range``
    * zero-volume bars            -- volume-weighted divisions
    * a constant-price run        -- zero standard deviation / zero range windows
    * a single extreme bar        -- whole-series min/max normalisation

    Because they all live past ``max(CUTS)``, a causal indicator can never let
    them influence a prefix result.
    """
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0, 1.0, bars))
    high = close + rng.uniform(0.1, 1.0, bars)
    low = close - rng.uniform(0.1, 1.0, bars)
    open_ = close + rng.normal(0, 0.3, bars)
    volume = rng.integers(1_000, 10_000, bars).astype(float)

    for i in (bars - 1, bars - 5, bars - 40):
        open_[i] = high[i] = low[i] = close[i]

    volume[bars - 3] = 0.0
    volume[bars - 12] = 0.0

    flat_run = slice(bars - 60, bars - 55)
    open_[flat_run] = high[flat_run] = low[flat_run] = close[flat_run] = close[bars - 60]

    high[bars - 25] *= 5.0
    low[bars - 25] /= 5.0
    volume[bars - 25] *= 100.0

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.date_range("2020-01-01", periods=bars, freq="D"),
    )


def window_schedule(index: pd.Index) -> pd.Series:
    """A per-bar window length, for indicators that require one (``mavp``)."""
    cycle = [5, 10, 20]
    return pd.Series([cycle[i % len(cycle)] for i in range(len(index))], index=index, dtype=float)


# Inputs an indicator needs but the accessor cannot derive from OHLCV.  This is
# test data, not policy: an indicator missing from here raises on its default
# call and fails its own case loudly rather than being skipped.
EXTRA_ARGS = {
    "mavp": lambda df: {"periods": window_schedule(df.index)},
}


def indicator_names() -> list[str]:
    return sorted({name for names in ta.Category.values() for name in names})


def boolean_kwargs(name: str) -> list[str]:
    """Names of the boolean parameters of indicator *name*."""
    func = getattr(ta, name, None)
    if func is None:
        return []
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return []

    found = []
    for parameter_name, parameter in signature.parameters.items():
        if parameter_name in SKIP_BOOL_KWARGS:
            continue
        annotation = parameter.annotation
        if annotation is inspect.Parameter.empty:
            continue
        optional_of = typing.get_args(annotation)
        if annotation is bool or (bool in optional_of and type(None) in optional_of):
            found.append(parameter_name)
    return found


def lookahead_reason(name: str, kwargs: dict):
    """Why this exact call is expected to look forward, or None."""
    rule = LOOKAHEAD_RULES.get(name)
    if rule is None:
        return None
    applies, reason = rule
    return reason if applies(kwargs) else None


def candidate_calls(name: str) -> list[dict]:
    """The default call plus each boolean keyword of *name* flipped on its own."""
    return [{}] + [{keyword: value} for keyword in boolean_kwargs(name) for value in (True, False)]


def first_non_causal_call(name: str) -> dict:
    """The simplest keyword set that puts *name* into its forward-looking mode.

    For dpo and ichimoku that is the plain default call; for cpr the mode is
    opt-in, so it is ``{"virgin_cpr": True}``.
    """
    for kwargs in candidate_calls(name):
        if lookahead_reason(name, kwargs):
            return kwargs
    raise AssertionError(f"{name} has a LOOKAHEAD_RULES entry but no forward-looking call")


def has_causal_mode(name: str) -> bool:
    """True when some keyword already switches the forward-looking branch off.

    Those indicators must also accept the library-wide `lookahead=False`; the
    ones with no causal mode at all (tos_stdevall, vp) have nothing to switch.
    """
    return any(lookahead_reason(name, kwargs) is None for kwargs in candidate_calls(name))


def sweep_cases():
    """Every (indicator, kwargs) call the sweep evaluates, marked where due."""
    cases = []
    for name in indicator_names():
        calls = [{}]
        calls += [{keyword: value} for keyword in boolean_kwargs(name) for value in (True, False)]
        for kwargs in calls:
            identifier = name + ("".join(f"[{k}={v}]" for k, v in kwargs.items()))
            reason = lookahead_reason(name, kwargs)
            marks = [pytest.mark.xfail(reason=reason, strict=True)] if reason else []
            cases.append(pytest.param(name, kwargs, id=identifier, marks=marks))
    return cases


def declined(result, source: pd.DataFrame) -> bool:
    """True when the accessor echoed the source frame instead of a result.

    An indicator that returns None because the prefix is shorter than its
    warm-up gets the input DataFrame back from the accessor; that is not a
    leak, just too little data.
    """
    return isinstance(result, pd.DataFrame) and list(result.columns) == list(source.columns)


def prefix_deviation(full, prefix):
    """Compare two results on their shared index; return a reason or None.

    Values are required to match exactly: a causal indicator replays the same
    operations on the same inputs for the shared rows, so any non-zero delta is
    a leak rather than accumulated floating point noise.
    """
    common = full.index.intersection(prefix.index)
    if len(common) == 0:
        return "no overlapping index"

    a = np.asarray(full.loc[common], dtype=float)
    b = np.asarray(prefix.loc[common], dtype=float)
    if a.shape != b.shape:
        return f"shape {a.shape} != {b.shape}"

    nan_mismatch = int((np.isnan(a) != np.isnan(b)).sum())
    if nan_mismatch:
        return f"{nan_mismatch} NaN placement mismatches"

    finite = ~np.isnan(a)
    if finite.any():
        delta = np.abs(a[finite] - b[finite]).max()
        if delta > 0:
            return f"max deviation {delta:.3e}"
    return None


@pytest.fixture(scope="module")
def frames():
    """The full frame plus one prefix per cut point."""
    df = causal_test_data()
    return df, {cut: df.head(cut).copy() for cut in CUTS}


_COLUMN_FOR_PARAM = {"open_": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}


def direct_call(source: pd.DataFrame, name: str, kwargs: dict):
    """Call the plain indicator function, bypassing the DataFrame accessor.

    The accessor turns a None result into the source frame
    (AnalysisIndicators._post_process), so the function is the only place a
    None contract can be observed.
    """
    func = getattr(ta, name)
    columns = {p: source[c] for p, c in _COLUMN_FOR_PARAM.items() if p in inspect.signature(func).parameters}
    return func(**columns, **kwargs)


def call(source: pd.DataFrame, name: str, kwargs: dict):
    extra = EXTRA_ARGS.get(name)
    return getattr(source.ta, name)(**{**(extra(source) if extra else {}), **kwargs})


def deviations(frames, name: str, kwargs: dict) -> list[str]:
    """Every cut point at which *name* disagrees with its own prefix."""
    df, prefixes = frames
    found = []
    full = call(df, name, kwargs)
    if full is None or declined(full, df):
        return found
    for cut, source in prefixes.items():
        prefix = call(source, name, kwargs)
        if prefix is None or declined(prefix, source):
            continue
        deviation = prefix_deviation(full, prefix)
        if deviation is not None:
            found.append(f"cut={cut}: {deviation}")
    return found


# --- the sweep -------------------------------------------------------------


@pytest.mark.parametrize("name, kwargs", sweep_cases())
def test_indicator_is_causal(frames, name, kwargs):
    """The first K bars must not change when later bars are appended.

    Calls marked xfail here look forward on purpose; the mark is strict, so one
    that becomes causal fails until its LOOKAHEAD_RULES entry is narrowed.
    """
    assert deviations(frames, name, kwargs) == []


@pytest.mark.parametrize("name", sorted(LOOKAHEAD_RULES))
def test_lookahead_false_is_all_or_nothing(frames, name):
    """Where `lookahead=False` is honoured it must yield a fully causal result.

    Compared against the indicator's forward-looking call, not its default:
    cpr's non-causal mode is opt-in, so comparing defaults would report that it
    honours the keyword when it does nothing at all.
    """
    df, _ = frames
    if not has_causal_mode(name):
        pytest.skip(f"{name} has no causal mode; see test_indicator_without_causal_mode_declines")

    kwargs = first_non_causal_call(name)
    baseline = call(df, name, kwargs)
    opted_out = call(df, name, {**kwargs, "lookahead": False})
    assert not baseline.equals(opted_out), f"{name} silently ignores lookahead=False"

    assert deviations(frames, name, {**kwargs, "lookahead": False}) == []


# --- issue #149: the epsilon must apply to the flat row only ---------------


def test_flat_bar_does_not_change_other_rows():
    high = pd.Series([10.0, 11.0, 12.0, 13.0])
    low = pd.Series([9.0, 10.0, 11.0, 13.0])

    result = non_zero_range(high, low)

    np.testing.assert_array_equal(result.iloc[:3].to_numpy(), np.array([1.0, 1.0, 1.0]))
    assert result.iloc[3] == sflt.epsilon


def test_appending_flat_bar_leaves_history_untouched():
    high = pd.Series([10.0, 11.0, 12.0])
    low = pd.Series([9.0, 10.0, 11.0])

    before = non_zero_range(high, low)
    after = non_zero_range(
        pd.concat([high, pd.Series([13.0], index=[3])]),
        pd.concat([low, pd.Series([13.0], index=[3])]),
    )

    pd.testing.assert_series_equal(before, after.iloc[:3])


def test_nan_rows_are_preserved():
    high = pd.Series([np.nan, 11.0, 12.0])
    low = pd.Series([np.nan, 11.0, 11.0])

    result = non_zero_range(high, low)

    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == sflt.epsilon
    assert result.iloc[2] == 1.0


@pytest.mark.parametrize("name", ["pdist", "brar", "true_range", "atr", "ad"])
def test_indicators_downstream_of_the_epsilon(frames, name):
    """The indicators named in issue #149, plus their closest neighbours."""
    assert deviations(frames, name, {}) == []


# --- the individually fixed cases ------------------------------------------


def test_mavp_refuses_to_invent_a_schedule(frames):
    """A `periods` default derived from len(close) would be a leak."""
    df, _ = frames
    with pytest.raises(TypeError):
        ta.mavp(df["close"])


def test_cdl_z_full_is_anchored(frames):
    """full=True is anchored, not whole-sample + bfill (see issue #149 follow-up)."""
    assert deviations(frames, "cdl_z", {"full": True}) == []


@pytest.mark.parametrize("name", sorted(n for n in LOOKAHEAD_RULES if not has_causal_mode(n)))
def test_indicator_without_causal_mode_declines(frames, name):
    """An indicator with nothing to switch off must refuse, not ignore the keyword.

    df.ta.strategy(..., lookahead=False) forwards the keyword to every
    indicator, so silently ignoring it hands a caller who asked for
    backtest-safe output a column that is not.
    """
    df, _ = frames
    with pytest.warns(UserWarning, match="no causal mode"):
        assert direct_call(df, name, {"lookahead": False}) is None


@pytest.mark.parametrize("name", sorted(n for n in LOOKAHEAD_RULES if not has_causal_mode(n)))
def test_accessor_appends_nothing_when_declining(frames, name):
    """Through the accessor the declined result must not add columns.

    The accessor hands back the source frame rather than None (see
    AnalysisIndicators._post_process); what matters here is that nothing
    forward-looking is appended.
    """
    df, _ = frames
    working = df.copy()
    before = list(working.columns)
    with pytest.warns(UserWarning, match="no causal mode"):
        getattr(working.ta, name)(lookahead=False, append=True)
    assert list(working.columns) == before


# --- the exemptions have to reach the reader too ----------------------------

DOCS = Path(__file__).resolve().parent.parent / "docs" / "indicators.rst"


@pytest.mark.parametrize("name", sorted(LOOKAHEAD_RULES))
def test_non_causal_indicator_is_documented(name):
    """Every xfail'd indicator must appear in the docs' causality section.

    Without this the reference page silently falls behind LOOKAHEAD_RULES, and a
    user planning a backtest has no way to find out which indicators peek ahead.
    """
    if not DOCS.is_file():
        pytest.skip(f"{DOCS} not available")
    text = DOCS.read_text(encoding="utf-8")
    section = text.partition("Lookahead Bias and Causality")[2]
    assert section, "docs/indicators.rst has no 'Lookahead Bias and Causality' section"
    assert f"``{name}(" in section, f"{name} is not listed in the docs' causality section"
