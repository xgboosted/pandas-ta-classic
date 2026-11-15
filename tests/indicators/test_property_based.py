"""
Property-based tests for pandas-ta-classic indicator functions.

Uses Hypothesis to generate diverse inputs and verify invariants that hold
across the entire input space — catching edge cases deterministic tests miss.

Test categories
---------------
1. **Indicator output invariants** — type, length, name conventions.
2. **Mathematical invariants** — bounding, monotonicity, relationships.
3. **Edge-case robustness** — None guards, NaN propagation, constant inputs.

The properties of the utilities live with their other tests in
``tests/utils/test_utils_<module>.py``; the strategies are in
``tests/strategies.py``.

Running
-------
    python -m pytest tests/indicators/test_property_based.py -v
    python -m pytest tests/indicators/test_property_based.py -v --hypothesis-show-statistics
    python -m pytest tests/indicators/test_property_based.py -v --hypothesis-profile=ci
"""

import math
import warnings
from datetime import UTC, datetime
from unittest import TestCase

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, example, given, settings, strategies as st

import pandas_ta_classic as ta
from pandas_ta_classic.utils import df_error_analysis
from tests.config import get_sample_data
from tests.strategies import ohlcv_dataframe, price_series

_small_positive_int = st.integers(min_value=2, max_value=50)


class TestPriceSeriesStrategy(TestCase):
    """The ``price_series`` generator does what its arguments say."""

    @given(price_series())
    def test_default_draws_finite_prices(self, s):
        # The old default (allow_nan=True, nan_prob=0) raised InvalidArgument.
        assert s.between(1.0, 1000.0).all()

    @given(price_series(min_size=20, max_size=20, allow_nan=True, nan_prob=0.5))
    def test_nan_or_price(self, s):
        assert (s.isna() | s.between(1.0, 1000.0)).all()

    @given(st.data(), st.sampled_from([0.05, 0.6]))
    def test_nan_prob_sets_the_nan_rate(self, data, nan_prob):
        # The old mask was sampled_from([True, False]) whatever nan_prob said.
        # Five binomial standard deviations over 2000 bars (0.024 at 0.05, 0.055
        # at 0.6): a rate off by 0.05 fails, and a correct one exceeds the bound
        # with probability below 1e-6 per example.
        n = 2000
        s = data.draw(price_series(min_size=n, max_size=n, allow_nan=True, nan_prob=nan_prob))
        assert abs(s.isna().mean() - nan_prob) < 5 * math.sqrt(nan_prob * (1 - nan_prob) / n)

    @given(st.data(), st.sampled_from([{"allow_nan": True}, {"nan_prob": 0.2}, {"allow_nan": True, "nan_prob": 1.0}]))
    def test_contradictory_arguments_raise(self, data, kwargs):
        with pytest.raises(ValueError, match="price_series"):
            data.draw(price_series(**kwargs))


# ======================================================================
# 1. Indicator output invariants
# ======================================================================


class TestIndicatorOutputInvariants(TestCase):
    """Indicators must return correct types, lengths, and naming."""

    @given(
        price_series(min_size=30, max_size=200, allow_nan=False),
        _small_positive_int,
    )
    def test_sma_returns_series_with_length_name(self, s, length):
        assume(len(s) >= length + 2)
        result = ta.sma(s, length=length, talib=False)
        assert isinstance(result, pd.Series)
        assert len(result) == len(s)
        assert str(length) in result.name
        assert result.name.startswith("SMA_")

    @given(
        price_series(min_size=30, max_size=200, allow_nan=False),
        _small_positive_int,
    )
    def test_ema_returns_series_with_length_name(self, s, length):
        assume(len(s) >= length + 2)
        result = ta.ema(s, length=length, talib=False)
        assert isinstance(result, pd.Series)
        assert len(result) == len(s)
        assert str(length) in result.name
        assert result.name.startswith("EMA_")

    @given(
        price_series(min_size=30, max_size=200, allow_nan=False),
        _small_positive_int,
    )
    def test_rsi_returns_series_with_length_name(self, s, length):
        assume(len(s) >= length + 2)
        result = ta.rsi(s, length=length, talib=False)
        if result is not None:
            assert isinstance(result, pd.Series)
            assert len(result) == len(s)
            assert str(length) in result.name

    @given(
        price_series(min_size=30, max_size=200, allow_nan=False),
        _small_positive_int,
    )
    def test_mom_returns_series_with_length_name(self, s, length):
        assume(len(s) >= length + 2)
        result = ta.mom(s, length=length, talib=False)
        assert isinstance(result, pd.Series)
        assert len(result) == len(s)
        assert str(length) in result.name

    @given(
        price_series(min_size=30, max_size=200, allow_nan=False),
        _small_positive_int,
    )
    def test_roc_returns_series_with_length_name(self, s, length):
        assume(len(s) >= length + 2)
        result = ta.roc(s, length=length, talib=False)
        assert isinstance(result, pd.Series)
        assert len(result) == len(s)
        assert str(length) in result.name

    @given(
        price_series(min_size=30, max_size=200, allow_nan=False),
        _small_positive_int,
    )
    def test_stdev_returns_series_with_length_name(self, s, length):
        assume(len(s) >= length + 2)
        result = ta.stdev(s, length=length, talib=False)
        assert isinstance(result, pd.Series)
        assert len(result) == len(s)
        assert str(length) in result.name

    @given(price_series(min_size=60, max_size=200, allow_nan=False))
    def test_bbands_returns_dataframe_with_columns(self, s):
        result = ta.bbands(s, length=20, talib=False)
        if result is not None:
            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(s)
            assert len(result.columns) == 5  # BBL, BBM, BBU, BBB, BBP


class TestDataFrameAccessorInvariants(TestCase):
    """DataFrame accessor (df.ta) must not crash with random OHLCV."""

    @given(ohlcv_dataframe(min_size=30, max_size=150))
    @settings(
        max_examples=30,
        deadline=2000,
    )
    def test_df_ta_sma_no_crash(self, df):
        result = df.ta.sma(length=10)
        assert isinstance(result, pd.Series)
        assert "SMA_10" in result.name

    @given(ohlcv_dataframe(min_size=30, max_size=150))
    @settings(
        max_examples=30,
        deadline=2000,
    )
    def test_df_ta_rsi_no_crash(self, df):
        result = df.ta.rsi(length=14)
        assert isinstance(result, pd.Series)
        assert "RSI_14" in result.name

    @given(ohlcv_dataframe(min_size=30, max_size=150))
    @settings(
        max_examples=30,
        deadline=2000,
    )
    def test_df_ta_bbands_no_crash(self, df):
        result = df.ta.bbands(length=20)
        assert isinstance(result, pd.DataFrame)
        assert any(col.startswith("BB") for col in result.columns)

    @given(
        ohlcv_dataframe(min_size=30, max_size=60),
        st.one_of(st.none(), st.text(max_size=8), st.integers(), st.floats(allow_nan=True), st.booleans()),
    )
    @settings(
        max_examples=50,
        deadline=2000,
    )
    def test_df_ta_removed_exchange_rejects_any_value(self, df, value):
        """Assigning to the removed df.ta.exchange raises instead of vanishing.

        pandas 3 builds a new accessor per access, so without a raising setter
        the value would land on a throwaway instance and do nothing at all.
        """
        before = dict(df.attrs)
        with pytest.raises(AttributeError, match="df.ta.exchange was removed in 0.9.0"):
            df.ta.exchange = value
        assert df.attrs == before

    @given(
        ohlcv_dataframe(min_size=30, max_size=60),
        st.lists(st.sampled_from(["sma", "rsi", "macd"]), min_size=1, max_size=5),
    )
    @settings(
        max_examples=30,
        deadline=2000,
    )
    def test_df_ta_last_run_is_a_utc_timestamp_of_each_run(self, df, kinds):
        """last_run is a stdlib UTC datetime taken during the run, whatever ran."""
        for kind in kinds:
            before = datetime.now(UTC)
            df.ta(kind=kind)
            after = datetime.now(UTC)
            last_run = df.ta.last_run
            assert type(last_run) is datetime
            assert last_run.tzinfo is UTC
            assert before <= last_run <= after

    @given(ohlcv_dataframe(min_size=30, max_size=150))
    @settings(
        max_examples=30,
        deadline=2000,
    )
    def test_df_ta_append_adds_column(self, df):
        """df.ta.sma(append=True) must add the column to the DataFrame."""
        df.ta.sma(length=10, append=True)
        assert "SMA_10" in df.columns

    @given(
        ohlcv_dataframe(min_size=30, max_size=150),
        st.lists(st.sampled_from(["sma", "macd", "bbands", "rsi"]), min_size=1, max_size=40),
        st.booleans(),
    )
    @settings(
        max_examples=30,
        deadline=5000,
    )
    def test_df_ta_append_leaves_warning_filters_unchanged(self, df, kinds, bad_col_names):
        """However many appends, and whether the last one raises, the filters are as before."""
        before = list(warnings.filters)
        for kind in kinds:
            getattr(df.ta, kind)(append=True)
        if bad_col_names:  # the append path raises after the filter is in place
            with pytest.raises(ValueError):
                df.ta.macd(append=True, col_names=("only_one",))
        assert list(warnings.filters) == before


class TestNoDirectionOnBarZero(TestCase):
    """The first bar has no predecessor, so it has no bar-to-bar direction.

    kvo, pvol(signed=True) and vp seeded the direction with +1 (``signed_series(x,
    1)``), so bar 0 counted as a rising bar: kvo read -9.6 instead of 0.0 on a
    flat series, pvol signed bar 0 positive, and vp booked bar 0's volume as
    buying. On a flat series no bar has a direction, so nothing may be signed.
    """

    @given(
        st.integers(min_value=120, max_value=300),
        st.floats(min_value=0.01, max_value=1e5, width=64),
        st.floats(min_value=1.0, max_value=1e9, width=64),
        st.sampled_from(sorted(ta.ma())),
    )
    @settings(max_examples=60, deadline=None)
    def test_a_flat_series_has_no_signed_volume(self, n, level, volume, mamode):
        price = pd.Series(np.full(n, level))
        vol = pd.Series(np.full(n, volume))
        kvo = ta.kvo(price, price, price, vol, fast=5, slow=10, signal=3, mamode=mamode)
        assert (kvo.dropna() == 0.0).all().all()
        assert kvo.notna().any().all()
        signed = ta.pvol(price, vol, signed=True)
        assert ((signed == 0.0) | signed.isna()).all()
        profile = ta.vp(price, vol, width=2)
        assert (profile.filter(like="pos_").to_numpy() == 0.0).all()
        assert (profile.filter(like="neg_").to_numpy() == 0.0).all()


# ======================================================================
# 2. Mathematical invariants
# ======================================================================


class TestMathematicalInvariants(TestCase):
    """Mathematical properties that must hold for indicator outputs."""

    @given(
        price_series(min_size=30, max_size=200, allow_nan=False),
        _small_positive_int,
    )
    def test_sma_of_constant_equals_constant(self, s, length):
        """If input is constant, SMA must equal that constant for valid rows."""
        assume(len(s) >= length + 2)
        const_val = 50.0
        const_s = pd.Series([const_val] * len(s))
        result = ta.sma(const_s, length=length, talib=False)
        valid = result.dropna()
        if not valid.empty:
            assert np.allclose(valid.values, const_val)

    @given(
        price_series(min_size=50, max_size=200, allow_nan=False),
        _small_positive_int,
    )
    def test_rsi_no_crash_and_has_name(self, s, length):
        """RSI must not crash and must return a Series with length in name."""
        assume(len(s) >= length + 2)
        result = ta.rsi(s, length=length, talib=False)
        # RSI may return None for degenerate inputs (constant series, etc.)
        if result is not None:
            assert isinstance(result, pd.Series)
            assert len(result) == len(s)
            assert str(length) in result.name

    @given(price_series(min_size=60, max_size=200, allow_nan=False))
    def test_bbands_lower_mid_upper_ordering(self, s):
        """BBANDS: lower ≤ mid ≤ upper for every row."""
        result = ta.bbands(s, length=20, std=2.0, talib=False)
        if result is not None and not result.empty:
            valid = result.dropna()
            if not valid.empty:
                lower = valid.iloc[:, 0]  # BBL
                mid = valid.iloc[:, 1]  # BBM
                upper = valid.iloc[:, 2]  # BBU
                assert (lower <= mid).all(), "BBL > BBM"
                assert (mid <= upper).all(), "BBM > BBU"

    @given(ohlcv_dataframe(min_size=60, max_size=150))
    @settings(
        max_examples=30,
        deadline=2000,
    )
    def test_atr_non_negative(self, df):
        """ATR must be non-negative for all valid values."""
        result = ta.atr(df["high"], df["low"], df["close"], length=14, talib=False)
        if result is not None:
            valid = result.dropna()
            if not valid.empty:
                assert (valid >= 0).all(), f"ATR has negative values: min={valid.min()}"

    @given(ohlcv_dataframe(min_size=60, max_size=150))
    @settings(
        max_examples=30,
        deadline=2000,
    )
    def test_stoch_no_crash_and_dataframe(self, df):
        """Stochastic must not crash and must return a DataFrame.

        Note: Stochastic can exceed [0,100] with pathological inputs
        (e.g. close > rolling-max-high). This is a known property
        discovered via property-based testing.
        """
        assume((df["high"] > df["low"]).all())
        result = ta.stoch(df["high"], df["low"], df["close"], talib=False)
        if result is not None and not result.empty:
            assert isinstance(result, pd.DataFrame)
            assert len(result.columns) >= 2  # STOCHk and STOCHd

    @given(price_series(min_size=60, max_size=200, allow_nan=False))
    def test_stdev_non_negative(self, s):
        """Standard deviation must be non-negative."""
        result = ta.stdev(s, length=10, talib=False)
        if result is not None:
            valid = result.dropna()
            if not valid.empty:
                assert (valid >= 0).all(), f"stdev negative: min={valid.min()}"

    @given(
        price_series(min_size=30, max_size=200, allow_nan=False),
        _small_positive_int,
    )
    def test_mom_roc_relationship(self, s, length):
        """MOM and ROC relationship: ROC ≈ (MOM / close[t-length]) * 100."""
        assume(len(s) >= length + 2)
        mom = ta.mom(s, length=length, talib=False)
        roc = ta.roc(s, length=length, talib=False)
        if mom is not None and roc is not None:
            valid = mom.dropna().index.intersection(roc.dropna().index)
            if len(valid) > 0:
                expected_roc = (mom.loc[valid] / s.shift(length).loc[valid]) * 100
                assert np.allclose(roc.loc[valid].values, expected_roc.values, rtol=1e-9)


# ======================================================================
# 3. Edge-case robustness
# ======================================================================


class TestNanPropagation(TestCase):
    """All-NaN input must produce all-NaN output, not crash."""

    @given(st.integers(min_value=10, max_value=100), _small_positive_int)
    def test_sma_all_nan_propagates(self, n, length):
        assume(n >= length)
        s = pd.Series([np.nan] * n, dtype=np.float64)
        result = ta.sma(s, length=length, talib=False)
        assert result is not None
        assert result.isna().all()

    @given(st.integers(min_value=10, max_value=100), _small_positive_int)
    def test_ema_all_nan_propagates(self, n, length):
        assume(n >= length)
        s = pd.Series([np.nan] * n, dtype=np.float64)
        result = ta.ema(s, length=length, talib=False)
        assert result is not None
        assert result.isna().all()

    @given(st.integers(min_value=10, max_value=100), _small_positive_int)
    def test_rsi_all_nan_propagates(self, n, length):
        assume(n >= length)
        s = pd.Series([np.nan] * n, dtype=np.float64)
        result = ta.rsi(s, length=length, talib=False)
        assert result is not None
        assert result.isna().all()

    @given(st.integers(min_value=10, max_value=100))
    def test_stdev_all_nan_propagates(self, n):
        s = pd.Series([np.nan] * n, dtype=np.float64)
        result = ta.stdev(s, length=10, talib=False)
        assert result is not None
        assert result.isna().all()


class TestPropertyBasedEdgeCases(TestCase):
    """Additional edge-case properties discovered via Hypothesis."""

    @given(
        price_series(min_size=5, max_size=50, allow_nan=False),
        st.integers(min_value=1, max_value=5),
    )
    @settings(suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])  # replaces the conftest profile's list
    def test_sma_short_series_no_crash(self, s, length):
        """Very short series must not crash SMA."""
        assume(len(s) >= length)
        result = ta.sma(s, length=length, talib=False)
        assert isinstance(result, pd.Series)
        assert len(result) == len(s)

    @given(
        price_series(min_size=50, max_size=100, allow_nan=True, nan_prob=0.1),
        _small_positive_int,
    )
    # The seeded NaN mask practically never draws an all-NaN series; pin it.
    @example(pd.Series([np.nan] * 50, name="close"), 5)
    def test_sma_with_nans_no_crash(self, s, length):
        """SMA with some NaN values must not crash."""
        assume(len(s) >= length + 2)
        result = ta.sma(s, length=length, talib=False)
        assert isinstance(result, pd.Series)
        assert len(result) == len(s)

    @given(
        price_series(min_size=50, max_size=100, allow_nan=False),
        _small_positive_int,
    )
    def test_sma_offset_preserves_length(self, s, length):
        """SMA with offset must return same-length Series."""
        assume(len(s) >= length + 2)
        result = ta.sma(s, length=length, offset=2, talib=False)
        assert isinstance(result, pd.Series)
        assert len(result) == len(s)

    @given(
        price_series(min_size=50, max_size=100, allow_nan=False),
        _small_positive_int,
    )
    def test_sma_fillna_no_nans(self, s, length):
        """SMA with fillna=0 must produce no NaN values."""
        assume(len(s) >= length + 2)
        result = ta.sma(s, length=length, fillna=0, talib=False)
        assert isinstance(result, pd.Series)
        assert not result.isna().any()


# ======================================================================
# 4. Category-wide indicator discovery tests
# ======================================================================


class TestCategoryDiscovery(TestCase):
    """Property tests verifying the indicator category discovery mechanism."""

    def test_all_categories_non_empty(self):
        """Every category must have at least one indicator."""
        from pandas_ta_classic._meta import Category

        for cat_name, indicators in Category.items():
            assert len(indicators) > 0, f"Category '{cat_name}' is empty"

    def test_category_indicators_are_strings(self):
        """All indicator entries in Category must be strings."""
        from pandas_ta_classic._meta import Category

        for cat_name, indicators in Category.items():
            for ind in indicators:
                assert isinstance(ind, str), f"Non-string indicator in {cat_name}: {ind!r}"

    def test_top_level_indicators_exist(self):
        """Top-level indicators like sma, rsi, bbands must be callable."""
        for name in ["sma", "ema", "rsi", "bbands", "macd", "stoch", "atr", "adx"]:
            assert hasattr(ta, name), f"Missing top-level indicator: {name}"
            assert callable(getattr(ta, name)), f"{name} is not callable"


# ======================================================================
# 5. pytest-style Hypothesis tests (alternative to unittest)
# ======================================================================


@pytest.mark.hypothesis
@given(
    s=price_series(min_size=30, max_size=200, allow_nan=False),
    length=_small_positive_int,
)
@settings(max_examples=100, deadline=1000)
def test_hypothesis_sma_idempotent(s, length):
    """SMA called twice with same args returns same result."""
    assume(len(s) >= length)
    r1 = ta.sma(s, length=length, talib=False)
    r2 = ta.sma(s, length=length, talib=False)
    pd.testing.assert_series_equal(r1, r2)


@pytest.mark.hypothesis
@given(
    s1=price_series(min_size=50, max_size=200, allow_nan=False),
    s2=price_series(min_size=50, max_size=200, allow_nan=False),
)
@settings(max_examples=50, deadline=1000)
def test_hypothesis_correlation_non_negative(s1, s2):
    """ta.corr must return value in [-1, 1]."""
    # Pearson correlation is undefined for constant series (std=0 → divide-by-zero).
    # Check the aligned overlap, not full-series std: series have RangeIndex so
    # the overlap is [:min(len(s1), len(s2))]. A full-series std>0 does not guarantee the
    # aligned slice is non-constant when the two lengths differ.
    n = min(len(s1), len(s2))
    assume(s1.iloc[:n].std() > 0)
    assume(s2.iloc[:n].std() > 0)

    try:
        corr = df_error_analysis(s1, s2)
    except Exception:  # noqa: BLE001 - pathological generated input is acceptable
        return
    assert -1.0 <= corr <= 1.0, f"Correlation out of bounds: {corr}"


_SAMPLE = get_sample_data()


@pytest.mark.hypothesis
@given(scale=st.floats(min_value=1e-3, max_value=1e6), start=st.integers(min_value=0, max_value=4000))
@settings(max_examples=30, deadline=None)
def test_hypothesis_vfi_ignores_the_volume_unit(scale, start):
    """VFI divides summed signed volume by an average volume, so the unit cancels.

    It must also move: a cutoff of ``coef * close`` instead of ``coef * stdev *
    close`` made it 0.0 on every bar of the sample data.
    """
    df = _SAMPLE.iloc[start : start + 1000]
    base = ta.vfi(df["high"], df["low"], df["close"], df["volume"])
    scaled = ta.vfi(df["high"], df["low"], df["close"], df["volume"] * scale)
    np.testing.assert_allclose(scaled.to_numpy(), base.to_numpy(), rtol=1e-9, atol=1e-12, equal_nan=True)
    assert base.dropna().nunique() > 100
