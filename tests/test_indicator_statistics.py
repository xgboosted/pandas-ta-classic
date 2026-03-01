from tests.config import (
    assert_columns,
    assert_nan_count,
    assert_offset,
    get_sample_data,
    CORRELATION,
    CORRELATION_THRESHOLD,
    HAS_TALIB,
    tal,
    talib_test,
    VERBOSE,
)
from tests.context import pandas_ta_classic as pandas_ta

from unittest import skip, TestCase
import numpy as np
import pandas.testing as pdt
from pandas import DataFrame, Series


class TestStatistics(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = get_sample_data()
        cls.data.columns = cls.data.columns.str.lower()
        cls.open = cls.data["open"]
        cls.high = cls.data["high"]
        cls.low = cls.data["low"]
        cls.close = cls.data["close"]
        if "volume" in cls.data.columns:
            cls.volume = cls.data["volume"]

    @classmethod
    def tearDownClass(cls):
        del cls.open
        del cls.high
        del cls.low
        del cls.close
        if hasattr(cls, "volume"):
            del cls.volume
        del cls.data

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_entropy(self):
        result = pandas_ta.entropy(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "ENTP_10")
        assert_offset(self, pandas_ta.entropy, self.close)
        assert_nan_count(self, result, 10)

    def test_kurtosis(self):
        result = pandas_ta.kurtosis(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "KURT_30")
        assert_offset(self, pandas_ta.kurtosis, self.close)
        assert_nan_count(self, result, 30)

    def test_mad(self):
        result = pandas_ta.mad(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "MAD_30")
        assert_offset(self, pandas_ta.mad, self.close)
        assert_nan_count(self, result, 30)

    def test_median(self):
        result = pandas_ta.median(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "MEDIAN_30")
        assert_offset(self, pandas_ta.median, self.close)
        assert_nan_count(self, result, 30)

    def test_quantile(self):
        result = pandas_ta.quantile(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "QTL_30_0.5")
        assert_offset(self, pandas_ta.quantile, self.close)
        assert_nan_count(self, result, 30)

    def test_skew(self):
        result = pandas_ta.skew(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "SKEW_30")
        assert_offset(self, pandas_ta.skew, self.close)
        assert_nan_count(self, result, 30)

    def test_stdev(self):
        result = pandas_ta.stdev(self.close, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "STDEV_30")

        result = pandas_ta.stdev(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "STDEV_30")
        assert_offset(self, pandas_ta.stdev, self.close, talib=False)
        stdev_result = pandas_ta.stdev(self.close, talib=False)
        assert_nan_count(self, stdev_result, 30)

    @talib_test
    def test_stdev_talib(self):
        result = pandas_ta.stdev(self.close, talib=False)
        expected = tal.STDDEV(self.close, 30)
        try:
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
            self.assertGreater(corr, CORRELATION_THRESHOLD)

    def test_tos_stdevall(self):
        result = pandas_ta.tos_stdevall(self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "TOS_STDEVALL")
        self.assertEqual(len(result.columns), 7)

        result = pandas_ta.tos_stdevall(self.close, length=30)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "TOS_STDEVALL_30")
        self.assertEqual(len(result.columns), 7)

        result = pandas_ta.tos_stdevall(self.close, length=30, stds=[1, 2])
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "TOS_STDEVALL_30")
        self.assertEqual(len(result.columns), 5)
        assert_columns(
            self,
            pandas_ta.tos_stdevall(self.close),
            [
                "TOS_STDEVALL_LR",
                "TOS_STDEVALL_L_1",
                "TOS_STDEVALL_U_1",
                "TOS_STDEVALL_L_2",
                "TOS_STDEVALL_U_2",
                "TOS_STDEVALL_L_3",
                "TOS_STDEVALL_U_3",
            ],
        )
        assert_offset(self, pandas_ta.tos_stdevall, self.close)

    def test_variance(self):
        result = pandas_ta.variance(self.close, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "VAR_30")

        result = pandas_ta.variance(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "VAR_30")
        assert_offset(self, pandas_ta.variance, self.close, talib=False)
        variance_result = pandas_ta.variance(self.close, talib=False)
        assert_nan_count(self, variance_result, 30)

    @talib_test
    def test_variance_talib(self):
        result = pandas_ta.variance(self.close, talib=False)
        expected = tal.VAR(self.close, 30)
        try:
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
            self.assertGreater(corr, CORRELATION_THRESHOLD)

    def test_kurtosis_vs_pandas(self):
        """Cross-validate numpy kurtosis against pandas rolling.kurt()."""
        result = pandas_ta.kurtosis(self.close)
        expected = self.close.rolling(30).kurt()
        mask = expected.notna()
        self.assertTrue(
            np.allclose(result[mask], expected[mask], atol=1e-7),
            "kurtosis deviates from pandas rolling.kurt()",
        )

    def test_skew_vs_pandas(self):
        """Cross-validate numpy skew against pandas rolling.skew()."""
        result = pandas_ta.skew(self.close)
        expected = self.close.rolling(30).skew()
        mask = expected.notna()
        self.assertTrue(
            np.allclose(result[mask], expected[mask], atol=1e-7),
            "skew deviates from pandas rolling.skew()",
        )

    def test_variance_vs_pandas(self):
        """Cross-validate numpy variance against pandas rolling.var()."""
        result = pandas_ta.variance(self.close, talib=False)
        expected = self.close.rolling(30).var()
        mask = expected.notna()
        self.assertTrue(
            np.allclose(result[mask], expected[mask], atol=1e-7),
            "variance deviates from pandas rolling.var()",
        )

    def test_stdev_vs_pandas(self):
        """Cross-validate numpy stdev against pandas rolling.std()."""
        result = pandas_ta.stdev(self.close, talib=False)
        expected = self.close.rolling(30).std()
        mask = expected.notna()
        self.assertTrue(
            np.allclose(result[mask], expected[mask], atol=1e-7),
            "stdev deviates from pandas rolling.std()",
        )

    def test_median_vs_pandas(self):
        """Cross-validate numpy median against pandas rolling.median()."""
        result = pandas_ta.median(self.close)
        expected = self.close.rolling(30).median()
        mask = expected.notna()
        self.assertTrue(
            np.allclose(result[mask], expected[mask], atol=1e-7),
            "median deviates from pandas rolling.median()",
        )

    def test_quantile_vs_pandas(self):
        """Cross-validate numpy quantile against pandas rolling.quantile()."""
        result = pandas_ta.quantile(self.close)
        expected = self.close.rolling(30).quantile(0.5)
        mask = expected.notna()
        self.assertTrue(
            np.allclose(result[mask], expected[mask], atol=1e-7),
            "quantile deviates from pandas rolling.quantile()",
        )

    def test_zscore(self):
        result = pandas_ta.zscore(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "ZS_30")
        assert_offset(self, pandas_ta.zscore, self.close)
        assert_nan_count(self, result, 30)
