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

import importlib
import inspect
from unittest import TestCase

import numpy as np
import pandas as pd
import pytest

import pandas_ta_classic as ta
from tests.assertions import assert_all_nan
from tests.config import get_sample_data
from tests.contracts import INDICATORS

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


@pytest.mark.parametrize("extra", ["osc", "ma1_ma2"])
def test_short_series_passed_through_kwargs_still_returns_all_nan(extra: str) -> None:
    """``stc`` takes ``ma1``/``ma2``/``osc`` through **kwargs.

    The shape-learning call must replace those too: keeping the caller's short
    series made it fail for the same reason the real call did, so the contract
    handed back None instead of an all-NaN result.
    """
    close = _short_frame()["close"]
    kwargs = {"osc": close} if extra == "osc" else {"ma1": close, "ma2": close}
    result = ta.stc(close, **kwargs)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["STC_10_12_26_0.5", "STCmacd_10_12_26_0.5", "STCstoch_10_12_26_0.5"]
    assert result.index.equals(close.index)
    assert result.isna().all().all()


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
    # the column is CDL_DOJI_10_0.1: checking for "CDL_DOJI_10" could never fail
    assert not any(column.startswith("CDL_DOJI_") for column in result.columns), list(result.columns)
    assert "CDL_INSIDE" in result.columns


def test_tables_cover_every_indicator() -> None:
    """Every indicator is classified, so a new one cannot slip in unnoticed.

    A new indicator fails this test until someone decides which contract it
    follows -- that decision is the point.
    """
    discovered = set(INDICATORS)
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


@pytest.mark.parametrize("length", [60_000, 600_000])
def test_window_longer_than_first_probe_still_returns_all_nan(length: int) -> None:
    """The shape-learning call was capped at 50,000 rows, so sma(length=60000) on 3 rows returned None."""
    close = _short_frame()["close"]
    result = ta.sma(close, length=length)
    assert isinstance(result, pd.Series)
    assert result.name == f"SMA_{length}"
    assert result.index.equals(close.index)
    assert result.isna().all()


# The two tests below each pin a `None` branch in utils/_core.py that a mutation
# run showed no test reached: removing the branch failed nothing, although the
# code would then call `.reindex` on None.


def test_inner_stage_too_short_for_its_window_returns_all_nan() -> None:
    """``_on_valid_rows`` passes on a None from its smoothing stage.

    On 15 rows the 14-bar raw stochastic has 2 valid bars, fewer than
    ``smooth_k=3``, so the smoothing stage returns None and the caller gets
    all NaN on its own index.
    """
    index = pd.date_range("2020-01-01", periods=15)
    close = pd.Series(np.linspace(100, 130, 15) + np.sin(np.arange(15)), index=index)
    result = ta.stoch(close + 1, close - 1, close, talib=False)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["STOCHk_14_3_3", "STOCHd_14_3_3"]
    assert result.index.equals(index)
    assert result.isna().all().all()


def test_leading_nan_run_leaving_too_few_rows_returns_all_nan() -> None:
    """``skip_leading_nan`` passes on a None from the trimmed call.

    25 leading NaN leave 10 rows for a 30-bar window: the trimmed call returns
    None, and the caller gets all NaN on its own index.
    """
    index = pd.date_range("2020-01-01", periods=35)
    close = pd.Series(np.r_[np.full(25, np.nan), np.arange(1.0, 11.0)], index=index)
    result = ta.maxindex(close, length=30)
    assert isinstance(result, pd.Series)
    assert result.name == "MAXINDEX_30"
    assert result.index.equals(index)
    assert result.isna().all()


# ---------------------------------------------------------------------------
# short-window hardening for rma, linreg and _sliding_weighted_ma
# ---------------------------------------------------------------------------


class TestShortWindowHardening(TestCase):
    """Three sites that crashed when handed a window longer than the series.

    None of them is reachable through the public API: verify_series' min_length
    guard blocks every route found while probing (oversized single and multi
    window arguments, every mamode, the accessor and strategy paths, NaN-prefixed
    input, and chained calls). They are hardened regardless, because the ema
    seeding bug (TestEmaSeedBounds in indicators/test_overlap.py) was equally unreachable until someone worked out which
    composition of indicators exposed it -- min_length models a single window,
    while real lookback is compositional.

    Reaching the guarded branches therefore requires bypassing verify_series,
    which is what these tests do.
    """

    @staticmethod
    def _passthrough(series, min_length=None):
        """verify_series without the min_length short-circuit."""
        return series if isinstance(series, pd.Series) else None

    def _without_guard(self, dotted_name):
        """Swap verify_series in a module for the pass-through, restoring after.

        The module has to be resolved by name: ``pandas_ta_classic.overlap``
        re-exports each indicator function under the same name as its submodule,
        so plain attribute access returns the function, not the module.
        """
        module = importlib.import_module(dotted_name)
        original = module.verify_series
        module.verify_series = self._passthrough
        self.addCleanup(setattr, module, "verify_series", original)
        return module

    def test_sliding_weighted_ma_window_longer_than_series(self):
        """The helper is directly callable and must not build an empty view."""
        from pandas_ta_classic.utils._core import _sliding_weighted_ma

        close = pd.Series([1.0, 2.0, 3.0])
        result = _sliding_weighted_ma(close, 10, np.ones(10))
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), 3)
        self.assertTrue(result.isna().all())

    def test_sliding_weighted_ma_exact_fit(self):
        """length == len(series) still produces the single valid window."""
        from pandas_ta_classic.utils._core import _sliding_weighted_ma

        close = pd.Series([1.0, 2.0, 3.0])
        result = _sliding_weighted_ma(close, 3, np.ones(3))
        self.assertEqual(result.iloc[-1], 6.0)
        self.assertTrue(result.iloc[:2].isna().all())

    def test_rma_without_guard(self):
        """rma seeded at iloc[length - 1] without checking it exists."""
        module = self._without_guard("pandas_ta_classic.overlap.rma")
        result = module.rma(pd.Series([1.0, 2.0, 3.0]), length=10)
        self.assertIsInstance(result, pd.Series)
        self.assertTrue(result.isna().all())

    def test_linreg_without_guard(self):
        """linreg built a sliding window wider than the input array."""
        module = self._without_guard("pandas_ta_classic.overlap.linreg")
        result = module.linreg(pd.Series([1.0, 2.0, 3.0]), length=10, talib=False)
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), 3)
        self.assertTrue(result.isna().all())

    def test_alma_without_guard(self):
        """alma reaches _sliding_weighted_ma, so it is covered too."""
        module = self._without_guard("pandas_ta_classic.overlap.alma")
        result = module.alma(pd.Series([1.0, 2.0, 3.0]), length=10)
        self.assertIsInstance(result, pd.Series)
        self.assertTrue(result.isna().all())

    def test_valid_input_unaffected(self):
        """Ample input keeps its usual lookback in all three."""
        close = pd.Series(np.arange(1.0, 101.0))
        self.assertEqual(ta.rma(close, length=10).isna().sum(), 9)
        self.assertEqual(ta.linreg(close, length=10, talib=False).isna().sum(), 9)
        self.assertEqual(ta.alma(close, length=10).isna().sum(), 9)


# ---------------------------------------------------------------------------
# short-input crashes where None reached arithmetic (issue #145)
# ---------------------------------------------------------------------------


class TestShortInputNoneNotPropagatedIntoArithmetic(TestCase):
    """A helper returned None for short input and the caller multiplied it.

    Found by sweeping every indicator over 1-400 rows with length, drift, fast,
    slow and signal at 1-5: pvt (drift > rows), rvi (length=1 on < 30 rows,
    because variance widens length=1 to its default window) and swma
    (length=1, symmetric_triangle(1) was None) raised TypeError; tos_stdevall
    on one row raised LinAlgError from a one-point linear fit.
    """

    @classmethod
    def setUpClass(cls):
        cls.df = get_sample_data().iloc[:200]

    def test_pvt_drift_longer_than_input(self):
        assert_all_nan(self, ta.pvt(self.df.close.iloc[:3], self.df.volume.iloc[:3], drift=5))

    def test_rvi_length_one_is_rejected(self):
        # variance used to widen length=1 to its default window; since 0.9.0
        # invalid window lengths raise instead of being replaced.
        d = self.df.iloc[:12]
        with self.assertRaisesRegex(ValueError, r"rvi\(\) length must be an integer > 1, got 1"):
            ta.rvi(d.close, d.high, d.low, length=1)

    def test_swma_length_one_is_the_input(self):
        np.testing.assert_allclose(ta.swma(self.df.close, length=1).to_numpy(), self.df.close.to_numpy())

    def test_symmetric_triangle_of_one(self):
        from pandas_ta_classic.utils import symmetric_triangle

        self.assertEqual(symmetric_triangle(1), [1])
        np.testing.assert_array_equal(symmetric_triangle(1, weighted=True), np.array([1.0]))

    def test_tos_stdevall_needs_two_points(self):
        assert_all_nan(self, ta.tos_stdevall(self.df.close.iloc[:1]))
        self.assertIsNotNone(ta.tos_stdevall(self.df.close.iloc[:2]))
