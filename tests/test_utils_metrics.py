import math
import warnings
from unittest import TestCase

import numpy as np
from pandas import DataFrame, Series, bdate_range

import pandas_ta_classic as pandas_ta
from tests.config import get_sample_data


class TestUtilityMetrics(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = get_sample_data()
        cls.close = cls.data["close"]
        cls.pctret = pandas_ta.percent_return(cls.close, cumulative=False)
        cls.logret = pandas_ta.percent_return(cls.close, cumulative=False)

    @classmethod
    def tearDownClass(cls):
        del cls.data
        del cls.pctret
        del cls.logret

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_cagr(self):
        result = pandas_ta.utils.cagr(self.data.close)
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)

        # Round trip: a price series that doubles over exactly one calendar
        # year must give a CAGR of ~100%.  The old calendar-days/252 bug made
        # this ~61%.
        idx = bdate_range("2021-01-04", "2022-01-04")
        close = Series(100 * 2.0 ** ((idx - idx[0]).days / 365.0), index=idx)
        self.assertAlmostEqual(pandas_ta.utils.cagr(close), 1.0, places=2)

    def test_calmar_ratio(self):
        result = pandas_ta.calmar_ratio(self.close)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)

        for bad_years in (0, -2):
            with self.subTest(years=bad_years), self.assertRaisesRegex(ValueError, "years must be an integer"):
                pandas_ta.calmar_ratio(self.close, years=bad_years)

        with self.assertRaisesRegex(ValueError, r"calmar_ratio\(\) method must be one of"):
            pandas_ta.calmar_ratio(self.close, method="bogus")

    def test_downside_deviation(self):
        result = pandas_ta.downside_deviation(self.pctret)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)

        result = pandas_ta.downside_deviation(self.logret)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)

    def test_drawdown(self):
        result = pandas_ta.drawdown(self.pctret)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "DD")

        result = pandas_ta.drawdown(self.logret)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "DD")

    def test_jensens_alpha(self):
        bench_return = self.pctret.sample(n=self.close.shape[0], random_state=1)
        result = pandas_ta.jensens_alpha(self.close, bench_return)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)

        # Constructed fixture: returns = 0.0005 + 1.2 * bench.  The benchmark
        # sums to less than 1 in absolute value, which the old ``int(x.sum())
        # != 0`` guard skipped, returning NaN instead of the intercept 0.0005.
        bench = Series(np.linspace(-0.002, 0.002, 200))
        returns = 0.0005 + 1.2 * bench
        self.assertAlmostEqual(pandas_ta.jensens_alpha(returns, bench), 0.0005, places=6)

    def test_log_max_drawdown(self):
        result = pandas_ta.log_max_drawdown(self.close)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)

    def test_max_drawdown(self):
        result = pandas_ta.max_drawdown(self.close)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)

        result = pandas_ta.max_drawdown(self.close, method="percent")
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)

        result = pandas_ta.max_drawdown(self.close, method="log")
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)

        result = pandas_ta.max_drawdown(self.close, all_methods=True)
        self.assertIsInstance(result, dict)
        self.assertIsInstance(result["dollar"], float)
        self.assertIsInstance(result["percent"], float)
        self.assertIsInstance(result["log"], float)

        with self.assertRaisesRegex(ValueError, r"method must be one of"):
            pandas_ta.max_drawdown(self.close, method="bogus")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = pandas_ta.max_drawdown(self.close, all=True)
            self.assertIsInstance(result, dict)
        self.assertTrue(any("deprecated" in str(x.message) for x in w))

    def test_optimal_leverage(self):
        result = pandas_ta.optimal_leverage(self.close)
        self.assertIsInstance(result, float)
        self.assertTrue(math.isfinite(result))
        result = pandas_ta.optimal_leverage(self.close, log=True)
        self.assertIsInstance(result, float)
        self.assertTrue(math.isfinite(result))
        constant = Series([100.0] * 60)
        with self.assertRaises(ValueError):
            pandas_ta.optimal_leverage(constant)

    def test_pure_profit_score(self):
        result = pandas_ta.pure_profit_score(self.close)
        self.assertGreaterEqual(result, 0)

        # A strictly linear series correlates r = 1.0 with its time index, so
        # the score equals the CAGR.  The old constant-zeros time index made the
        # correlation NaN and the function always returned 0.
        idx = bdate_range("2021-01-04", periods=250)
        linear = Series(100.0 + np.arange(250), index=idx)
        self.assertAlmostEqual(
            pandas_ta.pure_profit_score(linear),
            pandas_ta.cagr(linear),
            places=8,
        )

    def test_sharpe_ratio(self):
        result = pandas_ta.sharpe_ratio(self.close)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)

        result_cagr = pandas_ta.sharpe_ratio(self.close, use_cagr=True)
        self.assertIsInstance(result_cagr, float)
        self.assertGreaterEqual(result_cagr, 0)
        self.assertTrue(math.isfinite(result_cagr))

        result_log = pandas_ta.sharpe_ratio(self.close, use_cagr=True, log=True)
        self.assertIsInstance(result_log, float)
        self.assertGreaterEqual(result_log, 0)
        self.assertTrue(math.isfinite(result_log))

    def test_sortino_ratio(self):
        result = pandas_ta.sortino_ratio(self.close)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)

    def test_volatility(self):
        returns_ = pandas_ta.percent_return(self.close)
        result = pandas_ta.utils.volatility(returns_, returns=True)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)

        # Annualised daily volatility must land near σ·√252.  The old
        # calendar-days/252 bug understated it by ~17% (√173.6 vs √252).
        expected = float(returns_.std() * np.sqrt(252))
        self.assertAlmostEqual(result, expected, delta=expected * 0.02)

        for tf in ["years", "months", "weeks", "days", "hours", "minutes", "seconds"]:
            result = pandas_ta.utils.volatility(self.close, tf)
            with self.subTest(tf=tf):
                self.assertIsInstance(result, float)
                self.assertGreaterEqual(result, 0)
