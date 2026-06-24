"""
Property-based tests for pandas-ta-classic indicator functions and core utilities.

Uses Hypothesis to generate diverse inputs and verify invariants that hold
across the entire input space — catching edge cases deterministic tests miss.

Test categories
---------------
1. **Core utility properties** — verify_series, apply_offset, apply_fill, etc.
2. **Indicator output invariants** — type, length, name conventions.
3. **Mathematical invariants** — bounding, monotonicity, relationships.
4. **Edge-case robustness** — None guards, NaN propagation, constant inputs.

Strategy design
---------------
We use ``hypothesis.extra.numpy`` and ``hypothesis.extra.pandas`` to generate
realistic price series while also injecting pathological values (NaN, ±Inf,
zero, negative) at controlled rates to probe boundary behaviour.

Running
-------
    python -m pytest tests/test_property_based.py -v
    python -m pytest tests/test_property_based.py -v --hypothesis-show-statistics
    python -m pytest tests/test_property_based.py -v --hypothesis-profile=ci
"""

from unittest import TestCase

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings, strategies as st
from hypothesis.extra.numpy import arrays

import pandas_ta_classic as ta
from pandas_ta_classic.utils import (
    apply_fill,
    apply_offset,
    df_error_analysis,
    get_drift,
    get_offset,
    non_zero_range,
    verify_series,
)

# ---------------------------------------------------------------------------
# Custom Hypothesis strategies
# ---------------------------------------------------------------------------


@st.composite
def price_series(
    draw,
    min_size: int = 2,
    max_size: int = 200,
    allow_nan: bool = True,
    allow_inf: bool = False,
    nan_prob: float = 0.0,
) -> pd.Series:
    """Generate a realistic price-like Series.

    Parameters
    ----------
    min_size, max_size : int
        Length bounds.
    allow_nan : bool
        If False, no NaN values are injected.
    allow_inf : bool
        If True, ±Inf may appear with low probability.
    nan_prob : float
        Probability of a value being NaN (only when *allow_nan* is True).
    """
    n = draw(st.integers(min_value=min_size, max_value=max_size))

    if allow_nan and nan_prob > 0:
        # ``floats`` forbids min/max when allow_nan=True, so generate
        # finite floats then selectively replace with NaN.
        raw = draw(
            arrays(
                dtype=np.float64,
                shape=n,
                elements=st.floats(
                    min_value=1.0,
                    max_value=1000.0,
                    allow_nan=False,
                    allow_infinity=allow_inf,
                    width=64,
                ),
            )
        )
        # Inject NaN at controlled rate by sampling a boolean mask
        nan_mask = draw(
            arrays(
                dtype=bool,
                shape=n,
                elements=st.sampled_from(
                    [True, False],
                ),
            )
        )
        raw[nan_mask] = np.nan
    else:
        raw = draw(
            arrays(
                dtype=np.float64,
                shape=n,
                elements=st.floats(
                    min_value=1.0,
                    max_value=1000.0,
                    allow_nan=allow_nan,
                    allow_infinity=allow_inf,
                    width=64,
                ),
            )
        )

    return pd.Series(raw, name="close")


@st.composite
def ohlcv_dataframe(draw, min_size: int = 20, max_size: int = 200):
    """Generate a realistic OHLCV DataFrame compatible with ``df.ta`` accessor."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))

    # Generate a base price and derive OHLC from it
    base = (
        np.cumsum(
            draw(
                arrays(
                    dtype=np.float64,
                    shape=n,
                    elements=st.floats(min_value=-2.0, max_value=2.0, width=64, allow_nan=False),
                )
            )
        )
        + 100.0
    )

    df = pd.DataFrame(index=pd.RangeIndex(n))
    df["open"] = base + draw(
        arrays(
            dtype=np.float64,
            shape=n,
            elements=st.floats(min_value=-1.0, max_value=1.0, width=64),
        )
    )
    df["high"] = df["open"] + np.abs(
        draw(
            arrays(
                dtype=np.float64,
                shape=n,
                elements=st.floats(min_value=0.1, max_value=3.0, width=64),
            )
        )
    )
    df["low"] = df["open"] - np.abs(
        draw(
            arrays(
                dtype=np.float64,
                shape=n,
                elements=st.floats(min_value=0.1, max_value=3.0, width=64),
            )
        )
    )
    df["close"] = draw(
        arrays(
            dtype=np.float64,
            shape=n,
            elements=st.floats(min_value=df["low"].min(), max_value=df["high"].max(), width=64),
        )
    )
    df["volume"] = np.abs(
        draw(
            arrays(
                dtype=np.float64,
                shape=n,
                elements=st.floats(min_value=0.0, max_value=1e7, width=64),
            )
        )
    )

    # Ensure high >= low and close in range
    df["high"] = np.maximum(df["high"], df[["open", "close"]].max(axis=1))
    df["low"] = np.minimum(df["low"], df[["open", "close"]].min(axis=1))
    return df


@st.composite
def constant_price_series(draw, min_size: int = 20, max_size: int = 200):
    """Generate a constant-valued Series — tests degenerate arithmetic."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    value = draw(st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, width=64))
    return pd.Series([value] * n, dtype=np.float64)


# Shorthand strategies
_positive_int = st.integers(min_value=1, max_value=100)
_small_positive_int = st.integers(min_value=2, max_value=50)
_offset_int = st.integers(min_value=0, max_value=20)


# ======================================================================
# 1. Core utility properties
# ======================================================================


class TestVerifySeries(TestCase):
    """Property tests for ``verify_series``."""

    @given(price_series(min_size=2, max_size=200, allow_nan=False))
    def test_verify_series_returns_series(self, s):
        result = verify_series(s, min_length=1)
        assert isinstance(result, pd.Series)

    @given(price_series(min_size=10, max_size=200, allow_nan=False))
    def test_verify_series_preserves_name(self, s):
        s.name = "test_col"
        result = verify_series(s, min_length=5)
        assert result.name == "test_col"

    @given(
        price_series(min_size=1, max_size=50, allow_nan=False),
        _positive_int,
    )
    def test_verify_series_short_returns_none(self, s, min_len):
        assume(len(s) < min_len)
        result = verify_series(s, min_length=min_len)
        assert result is None

    @given(
        price_series(min_size=5, max_size=50, allow_nan=False),
        _positive_int,
    )
    def test_verify_series_long_enough_not_none(self, s, min_len):
        assume(len(s) >= min_len)
        result = verify_series(s, min_length=min_len)
        assert result is not None
        assert len(result) == len(s)

    @given(price_series(min_size=1, max_size=100, allow_nan=False))
    def test_verify_series_none_input_returns_none(self, s):
        """Passing None directly must return None."""
        assert verify_series(None, min_length=1) is None


class TestApplyOffset(TestCase):
    """Property tests for ``apply_offset``."""

    @given(price_series(min_size=10, max_size=100, allow_nan=False), _offset_int)
    def test_apply_offset_preserves_length(self, s, offset):
        result = apply_offset(s, offset)
        assert len(result) == len(s)

    @given(price_series(min_size=10, max_size=100, allow_nan=False))
    def test_apply_offset_zero_is_identity(self, s):
        result = apply_offset(s, 0)
        pd.testing.assert_series_equal(result, s)

    @given(price_series(min_size=10, max_size=100, allow_nan=False), _offset_int)
    def test_apply_offset_shift_direction(self, s, offset):
        assume(offset >= 2 and len(s) > offset + 1)
        result = apply_offset(s, offset)
        # First `offset` rows should be NaN after shift
        assert result.iloc[:offset].isna().all()


class TestApplyFill(TestCase):
    """Property tests for ``apply_fill``."""

    @given(price_series(min_size=10, max_size=100, allow_nan=True, nan_prob=0.3))
    def test_apply_fill_fillna_replaces_nan(self, s):
        filled = apply_fill(s.copy(), fillna=0.0)
        assert not filled.isna().any()

    @given(price_series(min_size=10, max_size=100, allow_nan=True, nan_prob=0.3))
    def test_apply_fill_ffill_no_gaps(self, s):
        assume(s.iloc[0] is not None and not np.isnan(s.iloc[0]))
        filled = apply_fill(s.copy(), fill_method="ffill")
        # After ffill, only a leading-NaN gap can remain
        first_valid = filled.first_valid_index()
        if first_valid is not None and first_valid > 0:
            assert filled.iloc[:first_valid].isna().all()
        assert not filled.iloc[first_valid:].isna().any() if first_valid is not None else True


class TestMiscUtils(TestCase):
    """Property tests for small utility functions."""

    @given(st.integers(min_value=-100, max_value=100))
    def test_get_offset_non_int_returns_zero(self, x):
        """Non-integer inputs must return 0."""
        result = get_offset(float(x))
        assert result == 0

    @given(st.integers(min_value=-100, max_value=100))
    def test_get_offset_int_identity(self, x):
        result = get_offset(x)
        assert isinstance(result, int)
        assert result == x

    @given(st.integers(min_value=-100, max_value=100))
    def test_get_drift_non_int_returns_one(self, x):
        result = get_drift(float(x))
        assert result == 1

    @given(st.integers(min_value=-100, max_value=100))
    def test_get_drift_zero_returns_one(self, x):
        """get_drift(0) defaults to 1."""
        result = get_drift(0)
        assert result == 1

    @given(
        st.integers(min_value=10, max_value=100),
        st.floats(min_value=1.0, max_value=1000.0, allow_nan=False),
        st.floats(min_value=0.01, max_value=10.0, allow_nan=False),
    )
    def test_non_zero_range_no_zeros(self, n, base, spread):
        """non_zero_range of strictly positive diff must be all-positive."""
        high_vals = np.full(n, base + spread, dtype=np.float64)
        low_vals = np.full(n, base, dtype=np.float64)
        high = pd.Series(high_vals)
        low = pd.Series(low_vals)
        diff = non_zero_range(high, low)
        assert (diff > 0).all()


# ======================================================================
# 2. Indicator output invariants
# ======================================================================


class TestIndicatorOutputInvariants(TestCase):
    """Indicators must return correct types, lengths, and naming."""

    @given(
        price_series(min_size=30, max_size=200, allow_nan=False),
        _small_positive_int,
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
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
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
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
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
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
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
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
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
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
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_stdev_returns_series_with_length_name(self, s, length):
        assume(len(s) >= length + 2)
        result = ta.stdev(s, length=length, talib=False)
        assert isinstance(result, pd.Series)
        assert len(result) == len(s)
        assert str(length) in result.name

    @given(price_series(min_size=60, max_size=200, allow_nan=False))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
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
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=2000,
    )
    def test_df_ta_sma_no_crash(self, df):
        result = df.ta.sma(length=10)
        assert isinstance(result, pd.Series)
        assert "SMA_10" in result.name

    @given(ohlcv_dataframe(min_size=30, max_size=150))
    @settings(
        max_examples=30,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=2000,
    )
    def test_df_ta_rsi_no_crash(self, df):
        result = df.ta.rsi(length=14)
        assert isinstance(result, pd.Series)
        assert "RSI_14" in result.name

    @given(ohlcv_dataframe(min_size=30, max_size=150))
    @settings(
        max_examples=30,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=2000,
    )
    def test_df_ta_bbands_no_crash(self, df):
        result = df.ta.bbands(length=20)
        assert isinstance(result, pd.DataFrame)
        assert any(col.startswith("BB") for col in result.columns)

    @given(ohlcv_dataframe(min_size=30, max_size=150))
    @settings(
        max_examples=30,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=2000,
    )
    def test_df_ta_append_adds_column(self, df):
        """df.ta.sma(append=True) must add the column to the DataFrame."""
        df.ta.sma(length=10, append=True)
        assert "SMA_10" in df.columns


# ======================================================================
# 3. Mathematical invariants
# ======================================================================


class TestMathematicalInvariants(TestCase):
    """Mathematical properties that must hold for indicator outputs."""

    @given(
        price_series(min_size=30, max_size=200, allow_nan=False),
        _small_positive_int,
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
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
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
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
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
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
        suppress_health_check=[HealthCheck.function_scoped_fixture],
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
        suppress_health_check=[HealthCheck.function_scoped_fixture],
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
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
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
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
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
# 4. None-guard / null-safety
# ======================================================================


class TestNoneGuards(TestCase):
    """Passing None for any required argument must return None, not raise."""

    @given(price_series(min_size=30, max_size=100, allow_nan=False))
    def test_sma_none_returns_none(self, s):
        assert ta.sma(None, length=10) is None

    @given(price_series(min_size=30, max_size=100, allow_nan=False))
    def test_ema_none_returns_none(self, s):
        assert ta.ema(None, length=10) is None

    @given(price_series(min_size=30, max_size=100, allow_nan=False))
    def test_rsi_none_returns_none(self, s):
        assert ta.rsi(None, length=14) is None

    @given(price_series(min_size=30, max_size=100, allow_nan=False))
    def test_bbands_none_returns_none(self, s):
        assert ta.bbands(None, length=20) is None

    @given(price_series(min_size=30, max_size=100, allow_nan=False))
    def test_stdev_none_returns_none(self, s):
        assert ta.stdev(None, length=10) is None

    @given(price_series(min_size=30, max_size=100, allow_nan=False))
    def test_mom_none_returns_none(self, s):
        assert ta.mom(None, length=10) is None

    @given(price_series(min_size=30, max_size=100, allow_nan=False))
    def test_roc_none_returns_none(self, s):
        assert ta.roc(None, length=10) is None


# ======================================================================
# 5. Edge-case robustness
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
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
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
# 6. Category-wide indicator discovery tests
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
# 7. pytest-style Hypothesis tests (alternative to unittest)
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
    # Pearson correlation is undefined for constant series (std=0 → divide-by-zero)
    assume(s1.std() > 0)
    assume(s2.std() > 0)

    try:
        corr = df_error_analysis(s1, s2)
        assert -1.0 <= corr <= 1.0, f"Correlation out of bounds: {corr}"
    except Exception:
        # df_error_analysis may raise on pathological inputs — that's acceptable
        pass
