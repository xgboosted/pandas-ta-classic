from unittest import TestCase

from pandas import DataFrame, Series

import pandas_ta_classic as pandas_ta
from tests.assertions import (
    CORRELATION_THRESHOLD,
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
            assert_talib(
                self,
                result,
                talib.STDDEV(self.close, 30),
                correlation_threshold=CORRELATION_THRESHOLD,
            )

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
            assert_talib(
                self,
                result,
                talib.VAR(self.close, 30),
                correlation_threshold=CORRELATION_THRESHOLD,
            )

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
