from unittest import TestCase

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

import pandas_ta_classic as pandas_ta
from tests.assertions import (
    IndicatorSpec,
    assert_all_nan,
    assert_indicator_standard,
    assert_talib,
)
from tests.config import get_sample_data

try:
    import talib

    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False


class TestStatistics(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = get_sample_data()
        cls.data.columns = cls.data.columns.str.lower()
        cls.open = cls.data["open"]
        cls.high = cls.data["high"]
        cls.low = cls.data["low"]
        cls.close = cls.data["close"]
        cls.volume = cls.data["volume"]

    @classmethod
    def tearDownClass(cls):
        del cls.open, cls.high, cls.low, cls.close, cls.volume, cls.data

    def test_beta(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.beta,
                args=[self.close],
                expected_name="BETA_30",
                kwargs={"benchmark": self.high},
                length_override=20,
            ),
        )

    def test_correl(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.correl,
                args=[self.close],
                expected_name="CORREL_30",
                kwargs={"benchmark": self.high},
                length_override=20,
            ),
        )

    def test_beta_and_correl_require_a_benchmark(self):
        """Omitting it used to return None, like an ordinary short input.

        strategy("all") leaves both out unless a benchmark= is broadcast, so
        nothing depends on the silent None any more.
        """
        for func, name in ((pandas_ta.beta, "beta"), (pandas_ta.correl, "correl")):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, rf"{name}\(\) requires a 'benchmark' Series"):
                    func(self.close)
                with self.assertRaisesRegex(ValueError, rf"{name}\(\) requires a 'benchmark' Series"):
                    func(self.close, benchmark=None)
                # A benchmark that is merely too short is data, not a mistake,
                # so the short-input contract still applies.
                short = func(self.close, benchmark=self.high.iloc[:5])
                self.assertIsInstance(short, Series)
                self.assertTrue(short.isna().all())

    def test_entropy(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.entropy,
                args=[self.close],
                expected_name="ENTP_10",
                length_override=20,
            ),
        )

    def test_kurtosis(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.kurtosis,
                args=[self.close],
                expected_name="KURT_30",
                length_override=20,
            ),
        )

    def test_mad(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.mad,
                args=[self.close],
                expected_name="MAD_30",
                length_override=20,
            ),
        )

    def test_mad_min_periods(self):
        # The min_periods branch uses the mean (not the median): for the partial
        # window [1, 1, 1, 10] the mean is 3.25 and MAD = 3.375, while a median
        # (1.0) would give 2.25.  This pins the mean so that branch cannot be
        # silently swapped for a median.
        close = Series([1.0, 1.0, 1.0, 10.0, 10.0])
        result = pandas_ta.mad(close, length=5, min_periods=2)
        self.assertAlmostEqual(result.iloc[3], 3.375)

    def test_md(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.md,
                args=[self.close],
                expected_name="MD_30",
                length_override=20,
            ),
        )

    def test_median(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.median,
                args=[self.close],
                expected_name="MEDIAN_30",
                length_override=20,
            ),
        )

    def test_quantile(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.quantile,
                args=[self.close],
                expected_name="QTL_30_0.5",
                length_override=20,
            ),
        )

    def test_skew(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.skew,
                args=[self.close],
                expected_name="SKEW_30",
                length_override=20,
            ),
        )

    def test_stderr(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.stderr,
                args=[self.close],
                expected_name="STDERR_14",
                length_override=20,
            ),
        )

    def test_stdev(self):
        result = assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.stdev,
                args=[self.close],
                expected_name="STDEV_30",
                kwargs={"talib": False},
                length_override=20,
            ),
        )
        if HAS_TALIB:
            assert_talib(self, result, talib.STDDEV(self.close, 30))

    def test_tos_stdevall(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.tos_stdevall,
                args=[self.close],
                expected_name="TOS_STDEVALL",
                expected_type=DataFrame,
                expected_columns=[
                    "TOS_STDEVALL_LR",
                    "TOS_STDEVALL_L_1",
                    "TOS_STDEVALL_U_1",
                    "TOS_STDEVALL_L_2",
                    "TOS_STDEVALL_U_2",
                    "TOS_STDEVALL_L_3",
                    "TOS_STDEVALL_U_3",
                ],
                length_override=30,
            ),
        )
        # stds unsorted → auto-reversed and still computes
        self.assertIsNotNone(pandas_ta.tos_stdevall(self.close, stds=[3, 2, 1]))
        # stds contains non-positive value → return None
        with self.assertRaisesRegex(ValueError, r"stds must all be > 0"):  # invalid argument, not short input
            pandas_ta.tos_stdevall(self.close, stds=[0, 1, 2])
        # length larger than data → verify_series returns None → return None
        assert_all_nan(self, pandas_ta.tos_stdevall(self.close.iloc[:5], length=30))

    def test_variance(self):
        result = assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.variance,
                args=[self.close],
                expected_name="VAR_30",
                kwargs={"talib": False},
                length_override=20,
            ),
        )
        if HAS_TALIB:
            assert_talib(self, result, talib.VAR(self.close, 30))

    def test_zscore(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.zscore,
                args=[self.close],
                expected_name="ZS_30",
                length_override=20,
            ),
        )

    def test_zscore_ddof(self):
        """ddof used to be missing, so cdl_z's forwarded value was dropped.

        The window deviation divides by length - ddof, so ddof=0 scales every
        value by sqrt(length / (length - 1)) against the ddof=1 default.
        """
        length = 30
        default = pandas_ta.zscore(self.close, length=length)
        sample = pandas_ta.zscore(self.close, length=length, ddof=1)
        population = pandas_ta.zscore(self.close, length=length, ddof=0)

        self.assertTrue(default.equals(sample), "the default must stay ddof=1")
        ratio = (population / sample).dropna()
        np.testing.assert_allclose(ratio.to_numpy(), np.sqrt(length / (length - 1)), rtol=1e-12)
        # A non-default ddof changes the values, so it is in the name (#343); the default keeps ZS_30.
        self.assertEqual(population.name, f"ZS_{length}_ddof0")
        self.assertEqual(sample.name, f"ZS_{length}")

        # ddof must leave a positive divisor, so it stays below length.
        with self.assertRaisesRegex(ValueError, r"zscore\(\) ddof must be an integer >= 0 and < 30"):
            pandas_ta.zscore(self.close, length=length, ddof=length)


# ---------------------------------------------------------------------------
# stdev / variance ddof default = 0 (population)
# ---------------------------------------------------------------------------


class TestStdevDdofDefault(TestCase):
    """stdev default ddof=0 (population std, not sample std)."""

    @classmethod
    def setUpClass(cls):
        cls.df = get_sample_data()
        cls.close = cls.df["close"]

    @classmethod
    def tearDownClass(cls):
        del cls.df
        del cls.close

    def test_default_matches_population_stdev(self):
        """ta.stdev() without ddof kwarg must equal pandas rolling std(ddof=0)."""
        length = 30
        result = pandas_ta.stdev(self.close, length=length, talib=False)
        expected = self.close.rolling(length).std(ddof=0)

        mask = result.notna() & expected.notna()
        self.assertTrue(mask.any(), "No overlapping non-NaN values to compare")
        np.testing.assert_allclose(
            result[mask].values,
            expected[mask].values,
            rtol=1e-4,
            err_msg="stdev() default ddof must be 0 (population std)",
        )

    def test_default_differs_from_sample_stdev(self):
        """ta.stdev() default must NOT equal pandas rolling std(ddof=1)."""
        length = 30
        result = pandas_ta.stdev(self.close, length=length, talib=False)
        sample_std = self.close.rolling(length).std(ddof=1)

        mask = result.notna() & sample_std.notna()
        diff = np.abs(result[mask].values - sample_std[mask].values)
        self.assertTrue(
            diff.max() > 1e-8,
            "stdev() default must NOT equal sample std (ddof=1); " "ddof=0 (population) is the correct post-fix default",
        )

    def test_explicit_ddof1_matches_sample_stdev(self):
        """ta.stdev(ddof=1) must equal pandas rolling std(ddof=1)."""
        length = 30
        result = pandas_ta.stdev(self.close, length=length, ddof=1, talib=False)
        expected = self.close.rolling(length).std(ddof=1)

        mask = result.notna() & expected.notna()
        self.assertTrue(mask.any())
        np.testing.assert_allclose(
            result[mask].values,
            expected[mask].values,
            rtol=1e-4,
            err_msg="stdev(ddof=1) must equal sample standard deviation",
        )


# ---------------------------------------------------------------------------
# zscore column name changed from Z_{length} to ZS_{length}
# ---------------------------------------------------------------------------


class TestZscoreColumnName(TestCase):
    """zscore output Series name must be ZS_{length}, not Z_{length}."""

    @classmethod
    def setUpClass(cls):
        cls.df = get_sample_data()
        cls.close = cls.df["close"]

    @classmethod
    def tearDownClass(cls):
        del cls.df
        del cls.close

    def test_zscore_column_name_prefix(self):
        """zscore() result name must start with 'ZS_', not 'Z_'."""
        length = 30
        result = pandas_ta.zscore(self.close, length=length)
        self.assertIsNotNone(result)
        self.assertTrue(
            result.name.startswith("ZS_"),
            f"zscore column name must start with 'ZS_'; got '{result.name}'",
        )

    def test_zscore_column_name_exact(self):
        """zscore(length=20) must produce column named 'ZS_20'."""
        result = pandas_ta.zscore(self.close, length=20)
        self.assertEqual(result.name, "ZS_20")

    def test_zscore_old_name_not_used(self):
        """zscore must not use old 'Z_{length}' naming convention."""
        result = pandas_ta.zscore(self.close, length=14)
        self.assertNotEqual(
            result.name,
            "Z_14",
            "zscore name 'Z_14' is the old pre-fix name; must be 'ZS_14'",
        )


# ---------------------------------------------------------------------------
# stdev forwards min_periods to variance
# ---------------------------------------------------------------------------


def test_stdev_forwards_min_periods():
    x = pd.Series([1.0, 4.0, 2.0, 8.0, 5.0, 7.0, 3.0, 9.0, 6.0, 10.0])
    got = pandas_ta.stdev(x, 5, min_periods=3)
    np.testing.assert_allclose(got, x.rolling(5, min_periods=3).std(ddof=0), equal_nan=True)
    assert np.isfinite(got.iloc[2])
