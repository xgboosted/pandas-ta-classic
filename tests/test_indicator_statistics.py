from tests.config import (
    assert_fill,
    assert_none_guard,
    assert_offset,
    error_analysis,
    get_sample_data,
    CORRELATION,
    CORRELATION_THRESHOLD,
    VERBOSE,
)
from tests.context import pandas_ta_classic as pandas_ta

from unittest import skip, TestCase
import pandas.testing as pdt
from pandas import DataFrame, Series

try:
    import talib as tal

    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    tal = None


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

    def test_beta(self):
        result = pandas_ta.beta(self.close, benchmark=self.high)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "BETA_30")
        pandas_ta.beta(self.close, benchmark=self.high, fillna=0)
        pandas_ta.beta(self.close, benchmark=self.high, fill_method="ffill")
        pandas_ta.beta(self.close, benchmark=self.high, fill_method="bfill")
        self.assertIsNone(pandas_ta.beta(None))
        assert_offset(self, pandas_ta.beta, [self.close], benchmark=self.high)
        assert_fill(self, pandas_ta.beta, [self.close], benchmark=self.high)

    def test_correl(self):
        result = pandas_ta.correl(self.close, benchmark=self.high)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "CORREL_30")
        pandas_ta.correl(self.close, benchmark=self.high, fillna=0)
        pandas_ta.correl(self.close, benchmark=self.high, fill_method="ffill")
        pandas_ta.correl(self.close, benchmark=self.high, fill_method="bfill")
        self.assertIsNone(pandas_ta.correl(None))
        assert_offset(self, pandas_ta.correl, [self.close], benchmark=self.high)
        assert_fill(self, pandas_ta.correl, [self.close], benchmark=self.high)

    def test_entropy(self):
        result = pandas_ta.entropy(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "ENTP_10")
        assert_offset(self, pandas_ta.entropy, [self.close])
        assert_fill(self, pandas_ta.entropy, [self.close])
        assert_none_guard(self, pandas_ta.entropy, [self.close])

    def test_kurtosis(self):
        result = pandas_ta.kurtosis(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "KURT_30")
        assert_offset(self, pandas_ta.kurtosis, [self.close])
        assert_fill(self, pandas_ta.kurtosis, [self.close])
        assert_none_guard(self, pandas_ta.kurtosis, [self.close])

    def test_mad(self):
        result = pandas_ta.mad(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "MAD_30")
        assert_offset(self, pandas_ta.mad, [self.close])
        assert_fill(self, pandas_ta.mad, [self.close])
        assert_none_guard(self, pandas_ta.mad, [self.close])

    def test_median(self):
        result = pandas_ta.median(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "MEDIAN_30")
        assert_offset(self, pandas_ta.median, [self.close])
        assert_fill(self, pandas_ta.median, [self.close])
        assert_none_guard(self, pandas_ta.median, [self.close])

    def test_quantile(self):
        result = pandas_ta.quantile(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "QTL_30_0.5")
        assert_offset(self, pandas_ta.quantile, [self.close])
        assert_fill(self, pandas_ta.quantile, [self.close])
        assert_none_guard(self, pandas_ta.quantile, [self.close])

    def test_skew(self):
        result = pandas_ta.skew(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "SKEW_30")
        assert_offset(self, pandas_ta.skew, [self.close])
        assert_fill(self, pandas_ta.skew, [self.close])
        assert_none_guard(self, pandas_ta.skew, [self.close])

    def test_stderr(self):
        result = pandas_ta.stderr(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "STDERR_14")
        self.assertIsNone(pandas_ta.stderr(None))
        assert_offset(self, pandas_ta.stderr, [self.close])
        assert_fill(self, pandas_ta.stderr, [self.close])
        assert_none_guard(self, pandas_ta.stderr, [self.close])

    def test_stdev(self):
        result = pandas_ta.stdev(self.close, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "STDEV_30")

        try:
            expected = tal.STDDEV(self.close, 30)
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            try:
                corr = pandas_ta.utils.df_error_analysis(
                    result, expected, col=CORRELATION
                )
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result, CORRELATION, ex)

        result = pandas_ta.stdev(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "STDEV_30")
        assert_offset(self, pandas_ta.stdev, [self.close])
        assert_fill(self, pandas_ta.stdev, [self.close])
        assert_none_guard(self, pandas_ta.stdev, [self.close])

    def test_tos_sdtevall(self):
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
        assert_offset(self, pandas_ta.tos_stdevall, [self.close])
        assert_fill(self, pandas_ta.tos_stdevall, [self.close])

    def test_variance(self):
        result = pandas_ta.variance(self.close, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "VAR_30")

        try:
            expected = tal.VAR(self.close, 30)
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            try:
                corr = pandas_ta.utils.df_error_analysis(
                    result, expected, col=CORRELATION
                )
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result, CORRELATION, ex)

        result = pandas_ta.variance(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "VAR_30")
        assert_offset(self, pandas_ta.variance, [self.close])
        assert_fill(self, pandas_ta.variance, [self.close])
        assert_none_guard(self, pandas_ta.variance, [self.close])

    def test_zscore(self):
        result = pandas_ta.zscore(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "ZS_30")
        assert_offset(self, pandas_ta.zscore, [self.close])
        assert_fill(self, pandas_ta.zscore, [self.close])
        assert_none_guard(self, pandas_ta.zscore, [self.close])
