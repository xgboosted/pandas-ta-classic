"""Characterization tests for indicator behaviour on below-minimum input.

These tests record what the library does TODAY when a Series is shorter than the
indicator's window. They are not an endorsement of that behaviour -- they exist so
that any future change to it shows up as an explicit, reviewable diff of these
tables rather than as a silent flip.

The documented contract lives in ``test_indicator_edge_cases.py``::

    When one OHLCV component is shorter than required (below minimum threshold)
    the indicator must return None rather than raise.

Two tables below pin the two observed outcomes on a 3-row OHLCV frame:

``RETURNS_NONE``
    The indicator's window exceeds the input length and ``verify_series`` was
    given a ``min_length``, so the guard fires and ``None`` comes back.

``RETURNS_SERIES``
    The indicator either has no window parameter (elementwise transforms such as
    ``hl2``, ``ohlc4``, ``bop``) or its default window is <= 3, so 3 rows is
    enough. Returning a Series here is correct.

    ``cdl_pattern`` belongs here because its native sub-patterns need only a few
    bars; the ones that need more (``doji``, window 10) are skipped rather than
    returned as columns.

Both tables reflect the state after the ``dm`` and ``cdl_pattern`` short-input
fixes. Before those, ``dm`` returned a DataFrame from 3 rows despite a default
``length`` of 14, and ``cdl_pattern`` raised ``AttributeError``.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

import pandas_ta_classic as ta

_N_SHORT = 3

# Series-valued parameters the harness knows how to synthesise.
_SERIES_PARAMS = frozenset({"open_", "open", "high", "low", "close", "volume", "benchmark"})

# Not covered: these take Series arguments that are outputs of other indicators
# rather than raw price data, so "below-minimum input" is not well defined for
# them without first deciding what to feed in.
_EXCLUDED = frozenset({"add", "div", "mult", "sub", "long_run", "short_run", "tsignals", "xsignals", "ma"})

# Opt out of the deprecated tuple return so the sweep stays warning-free.
_EXTRA_KWARGS = {"ichimoku": {"as_dataframe": False}}


def _frame_kwargs(name: str, frame: dict[str, pd.Series]) -> dict:
    """Extra arguments that have to be built from *frame* itself."""
    if name == "mavp":
        # `periods` carries one window length per bar. Supplying it explicitly
        # keeps the sweep independent of whatever default mavp happens to have:
        # the length schedule is the indicator's whole input, so leaving it to a
        # default would make this test assert on a moving target. A constant 10
        # stays above the rows handed in, so the short-input contract applies.
        return {"periods": pd.Series(10.0, index=frame["close"].index)}
    return {}


# Indicators that return ``None`` when handed fewer rows than their window.
RETURNS_NONE = frozenset(
    {
        "aberration",
        "accbands",
        "adosc",
        "adx",
        "adxr",
        "alma",
        "amat",
        "ao",
        "aobv",
        "apo",
        "aroon",
        "atr",
        "avolume",
        "bbands",
        "beta",
        "bias",
        "brar",
        "cci",
        "cdl_doji",
        "cdl_z",
        "ce",
        "cfo",
        "cg",
        "chop",
        "cksp",
        "cmf",
        "cmo",
        "coppock",
        "correl",
        "cti",
        "cvi",
        "decay",
        "dema",
        "donchian",
        "dm",
        "dpo",
        "dsp",
        "dx",
        "ebsw",
        "edecay",
        "efi",
        "ema",
        "entropy",
        "eom",
        "er",
        "eri",
        "fisher",
        "fosc",
        "fwma",
        "hilo",
        "hma",
        "hvol",
        "ichimoku",
        "inertia",
        "jma",
        "kama",
        "kc",
        "kdj",
        "kst",
        "kurtosis",
        "kvo",
        "linreg",
        "linregangle",
        "linregintercept",
        "linregslope",
        "lrsi",
        "macd",
        "macdext",
        "macdfix",
        "mad",
        "massi",
        "mavp",
        "maxindex",
        "mcgd",
        "md",
        "median",
        "mfi",
        "minindex",
        "minmax",
        "minmaxindex",
        "minus_dm",
        "mmar",
        "mom",
        "msw",
        "natr",
        "pgo",
        "plus_dm",
        "pmax",
        "po",
        "ppo",
        "psl",
        "pvo",
        "pwma",
        "qqe",
        "qstick",
        "quantile",
        "rainbow",
        "rma",
        "roc",
        "rocp",
        "rocr",
        "rocr100",
        "rolling_max",
        "rolling_min",
        "rolling_sum",
        "rsi",
        "rsx",
        "rvgi",
        "rvi",
        "sinwma",
        "skew",
        "sma",
        "smc_sweep",
        "smi",
        "squeeze",
        "squeeze_pro",
        "ssf",
        "stc",
        "stderr",
        "stdev",
        "stoch",
        "stochf",
        "stochrsi",
        "supertrend",
        "swma",
        "t3",
        "tema",
        "thermo",
        "trima",
        "trix",
        "trixh",
        "tsf",
        "tsi",
        "ttm_trend",
        "ui",
        "uo",
        "variance",
        "vfi",
        "vhf",
        "vidya",
        "vortex",
        "vosc",
        "vp",
        "vwma",
        "vwmacd",
        "willr",
        "wma",
        "zlma",
        "zscore",
    }
)

# Indicators that return a Series from a 3-row frame. Correct for all but ``dm``.
RETURNS_SERIES = frozenset(
    {
        "acos",
        "ad",
        "asin",
        "atan",
        "avgprice",
        "bop",
        "cdl_inside",
        "cdl_pattern",
        "ceil",
        "cos",
        "cosh",
        "cpr",
        "decreasing",
        "drawdown",
        "emv",
        "exp",
        "floor",
        "ha",
        "hl2",
        "hlc3",
        "ht_dcperiod",
        "ht_dcphase",
        "ht_phasor",
        "ht_sine",
        "ht_trendline",
        "ht_trendmode",
        "hwc",
        "hwma",
        "increasing",
        "ln",
        "log10",
        "log_return",
        "mama",
        "marketfi",
        "medprice",
        "midpoint",
        "midprice",
        "npabs",
        "npround",
        "nvi",
        "obv",
        "ohlc4",
        "pdist",
        "percent_return",
        "psar",
        "pvi",
        "pvol",
        "pvr",
        "pvt",
        "sarext",
        "sin",
        "sinh",
        "slope",
        "sqrt",
        "tan",
        "tanh",
        "td_seq",
        "todeg",
        "torad",
        "tos_stdevall",
        "true_range",
        "trunc",
        "typprice",
        "vwap",
        "wad",
        "wcp",
    }
)


def _short_frame(n: int = _N_SHORT) -> dict[str, pd.Series]:
    """A deterministic OHLCV frame with *n* rows -- below almost every window."""
    rng = np.random.default_rng(0)
    base = 100 + np.cumsum(rng.normal(0, 1, n))
    index = pd.date_range("2020-01-01", periods=n, freq="D")
    return {
        "open_": pd.Series(base - 0.5, index=index),
        "open": pd.Series(base - 0.5, index=index),
        "high": pd.Series(base + 1.0, index=index),
        "low": pd.Series(base - 1.0, index=index),
        "close": pd.Series(base, index=index),
        "volume": pd.Series(rng.integers(1_000, 5_000, n).astype(float), index=index),
        "benchmark": pd.Series(base * 0.99, index=index),
    }


def _call(name: str):
    """Call *name* with every price-like argument its signature accepts."""
    func = getattr(ta, name)
    frame = _short_frame()
    kwargs = {param: frame[param] for param in inspect.signature(func).parameters if param in _SERIES_PARAMS}
    result = func(**kwargs, **_EXTRA_KWARGS.get(name, {}), **_frame_kwargs(name, frame))
    # A few indicators (ichimoku) return a tuple; an all-None tuple is a None result.
    if isinstance(result, tuple) and all(part is None for part in result):
        return None
    return result


@pytest.mark.parametrize("name", sorted(RETURNS_NONE))
def test_short_input_returns_none(name: str) -> None:
    """Below-minimum input yields ``None`` -- the documented contract."""
    assert _call(name) is None, f"{name} no longer returns None on {_N_SHORT} rows"


@pytest.mark.parametrize("name", sorted(RETURNS_SERIES))
def test_short_input_returns_result(name: str) -> None:
    """No window (or a window <= 3 rows), so a real result comes back."""
    result = _call(name)
    assert result is not None, f"{name} now returns None on {_N_SHORT} rows"
    assert isinstance(result, (pd.Series, pd.DataFrame, tuple))


def test_cdl_pattern_skips_uncomputable_subpatterns() -> None:
    """``cdl_pattern`` drops sub-patterns it cannot compute rather than raising.

    ``cdl_doji`` needs 10 rows; on a 3-row frame it returns ``None`` and used to
    trigger ``AttributeError: 'NoneType' object has no attribute 'name'``.
    """
    result = _call("cdl_pattern")
    assert isinstance(result, pd.DataFrame)
    assert "CDL_DOJI_10" not in result.columns
    assert "CDL_INSIDE" in result.columns


def test_tables_cover_every_indicator() -> None:
    """Every indicator is classified, so a new one cannot slip in unnoticed.

    A new indicator fails this test until someone decides which contract it
    follows -- that decision is the point.
    """
    discovered = {
        name for names in ta.Category.values() for name in names if callable(getattr(ta, name, None)) and not inspect.isclass(getattr(ta, name))
    }
    classified = RETURNS_NONE | RETURNS_SERIES | _EXCLUDED
    unclassified = discovered - classified
    assert not unclassified, "unclassified indicator(s): " f"{sorted(unclassified)} -- add each to RETURNS_NONE or RETURNS_SERIES"
