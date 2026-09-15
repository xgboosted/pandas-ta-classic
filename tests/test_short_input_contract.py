"""Contract tests for indicator behaviour on below-minimum input (issue #145, case B).

When a Series is shorter than an indicator's window, the indicator returns an
all-NaN result with the input's index, the same way ``close.rolling(50).mean()``
on 10 rows returns 10 NaNs. Before 0.9.0 it returned ``None``, which silently
dropped the column in ``df.ta.<indicator>(append=True)``.

Two tables pin the two outcomes on a 3-row OHLCV frame:

``RETURNS_ALL_NAN``
    The indicator's window exceeds the input length. The result is all NaN,
    indexed like ``close`` and shaped like a normal result (same name and
    columns as a call on long input).

``RETURNS_SERIES``
    The indicator either has no window parameter (elementwise transforms such as
    ``hl2``, ``ohlc4``, ``bop``) or its default window is <= 3, so 3 rows is
    enough to compute real values.

    ``cdl_pattern`` belongs here because its native sub-patterns need only a few
    bars; the ones that need more (``doji``, window 10) are skipped rather than
    returned as columns.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

import pandas_ta_classic as ta

_N_SHORT = 3
_N_LONG = 400

# Series-valued parameters the harness knows how to synthesise.
_SERIES_PARAMS = frozenset({"open_", "open", "high", "low", "close", "volume", "benchmark"})

# Not covered: these take Series arguments that are outputs of other indicators
# rather than raw price data, so "below-minimum input" is not well defined for
# them without first deciding what to feed in.
_EXCLUDED = frozenset({"add", "div", "mult", "sub", "long_run", "short_run", "tsignals", "xsignals", "ma"})



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


# Indicators whose window exceeds a 3-row input: they return an all-NaN result.
RETURNS_ALL_NAN = frozenset(
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
    # Named like DataFrame columns, so outputs named after their inputs (vp's
    # low_close, pos_volume) are compared too.
    values = {
        "open_": base - 0.5,
        "open": base - 0.5,
        "high": base + 1.0,
        "low": base - 1.0,
        "close": base,
        "volume": rng.integers(1_000, 5_000, n).astype(float),
        "benchmark": base * 0.99,
    }
    return {key: pd.Series(v, index=index, name=key.rstrip("_")) for key, v in values.items()}


def _call(name: str, n: int = _N_SHORT):
    """Call *name* with every price-like argument its signature accepts."""
    func = getattr(ta, name)
    frame = _short_frame(n)
    kwargs = {param: frame[param] for param in inspect.signature(func).parameters if param in _SERIES_PARAMS}
    return func(**kwargs, **_frame_kwargs(name, frame))


@pytest.mark.parametrize("name", sorted(RETURNS_ALL_NAN))
def test_short_input_returns_all_nan(name: str) -> None:
    """Below-minimum input yields an all-NaN result on the input index, shaped like a normal result."""
    short = _call(name)
    assert isinstance(short, (pd.Series, pd.DataFrame)), f"{name} returned {type(short).__name__} on {_N_SHORT} rows"
    assert np.isnan(short.to_numpy(dtype=float)).all(), f"{name} returned values on {_N_SHORT} rows"
    full = _call(name, _N_LONG)
    # Row-aligned indicators keep the input index; binned ones (vp) keep their bin index.
    expected_index = _short_frame()["close"].index if len(full) == _N_LONG else full.index
    assert short.index.equals(expected_index)
    assert short.name == full.name
    if isinstance(full, pd.DataFrame):
        assert list(short.columns) == list(full.columns)


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
    classified = RETURNS_ALL_NAN | RETURNS_SERIES | _EXCLUDED
    unclassified = discovered - classified
    assert not unclassified, "unclassified indicator(s): " f"{sorted(unclassified)} -- add each to RETURNS_ALL_NAN or RETURNS_SERIES"


@pytest.mark.parametrize("name", sorted(RETURNS_ALL_NAN | RETURNS_SERIES))
def test_empty_input_returns_empty_result(name: str) -> None:
    """Zero rows is shorter than any window: an empty result shaped like a normal one, never a crash."""
    empty = _call(name, 0)
    full = _call(name, _N_LONG)
    assert isinstance(empty, (pd.Series, pd.DataFrame)), f"{name} returned {type(empty).__name__} on 0 rows"
    assert np.isnan(empty.to_numpy(dtype=float)).all()
    assert empty.name == full.name
    if isinstance(full, pd.DataFrame):
        assert list(empty.columns) == list(full.columns)
    if len(full) == _N_LONG:
        assert len(empty) == 0
