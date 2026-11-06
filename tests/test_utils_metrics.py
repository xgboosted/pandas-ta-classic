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

    def test_downside_deviation_counts_only_the_returns_it_sums(self):
        # The sum skipped NaN while the divisor counted it: the leading NaN of
        # pct_change gave sqrt(ss / n), a clean series sqrt(ss / (n - 1)), and
        # a gap inside a smaller value with no error.
        returns = self.pctret
        self.assertTrue(np.isnan(returns.iloc[0]))
        self.assertEqual(pandas_ta.downside_deviation(returns), pandas_ta.downside_deviation(returns.iloc[1:]))

        gappy = returns.copy()
        gappy.iloc[100] = np.nan
        with self.assertRaisesRegex(ValueError, r"downside_deviation\(\) returns has 1 missing value"):
            pandas_ta.downside_deviation(gappy)
        with self.assertRaisesRegex(ValueError, r"downside_deviation\(\) needs at least 2 returns, got 1"):
            pandas_ta.downside_deviation(returns.iloc[:2])

    def test_sortino_ratio_rejects_missing_close(self):
        gappy = self.close.copy()
        gappy.iloc[100] = np.nan
        with self.assertRaisesRegex(ValueError, r"sortino_ratio\(\) close has 1 missing value"):
            pandas_ta.sortino_ratio(gappy)

    def test_return_metrics_reject_missing_close(self):
        # A gap dropped two returns from the mean and the deviation without a
        # word: sharpe_ratio read 0.3305 instead of 0.3293 for one missing close.
        gappy = self.close.copy()
        gappy.iloc[2000] = np.nan
        for name in ("sharpe_ratio", "optimal_leverage", "volatility"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, rf"{name}\(\) close has 1 missing value"):
                getattr(pandas_ta.utils, name)(gappy)

    def test_volatility_does_not_count_the_first_bar(self):
        # The periods per year counted the first bar, which has no return, so
        # the result moved with whether the caller dropped it (0.198789 vs 0.198783).
        self.assertTrue(np.isnan(self.pctret.iloc[0]))
        with_nan = pandas_ta.utils.volatility(self.pctret, returns=True)
        self.assertEqual(with_nan, pandas_ta.utils.volatility(self.pctret.iloc[1:], returns=True))
        self.assertEqual(with_nan, pandas_ta.utils.volatility(self.close))

    def test_drawdown(self):
        # A return series is not a price: its percent and log drawdown read NaN
        # where it is at or below zero, with one warning (#345).
        for returns in (self.pctret, self.logret):
            with self.assertWarnsRegex(UserWarning, r"^drawdown\(\) reads NaN on \d+ bar\(s\) whose price is at or below zero"):
                result = pandas_ta.drawdown(returns)
            self.assertIsInstance(result, DataFrame)
            self.assertEqual(result.name, "DD")
            self.assertTrue(result["DD_PCT"][returns <= 0].isna().all())

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

        # the 'all' alias of all_methods was removed in 0.9.0
        with self.assertRaisesRegex(TypeError, "unexpected keyword argument 'all'"):
            pandas_ta.max_drawdown(self.close, all=True)

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

        # Annualised daily volatility must land near stddev * sqrt(252).  The old
        # calendar-days/252 bug understated it by ~17% (sqrt(173.6) vs sqrt(252)).
        expected = float(returns_.std() * np.sqrt(252))
        self.assertAlmostEqual(result, expected, delta=expected * 0.02)

        for tf in ["years", "months", "weeks", "days", "hours", "minutes", "seconds"]:
            result = pandas_ta.utils.volatility(self.close, tf)
            with self.subTest(tf=tf):
                self.assertIsInstance(result, float)
                self.assertGreaterEqual(result, 0)

        # nearest_day rounds the bars-per-year factor up, so it cannot lower the
        # annualised figure.
        yearly = pandas_ta.utils.volatility(self.close, "years")
        nearest = pandas_ta.utils.volatility(self.close, "years", nearest_day=True)
        self.assertGreaterEqual(nearest, yearly)

    def test_volatility_names_a_missing_close(self):
        # The other metrics are in test_every_utility_names_a_required_input_passed_as_none;
        # volatility is kept out of utils.__all__, so that sweep does not see it. All of them
        # answered None with NaN, a sentinel AGENTS.md rule 1 rules out for utils/_metrics.py.
        with self.assertRaisesRegex(ValueError, r"^volatility\(\) requires 'close', got None$"):
            pandas_ta.utils.volatility(None)

    def test_close_metrics_reject_missing_close(self):
        # A NaN on the first or last bar made cagr and log_max_drawdown read NaN (and
        # calmar_ratio for the last bar); one inside, or max_drawdown's anywhere, was
        # skipped without a word. The other close metrics already raised for any NaN.
        for position in (0, 2000, -1):
            gappy = self.close.copy()
            gappy.iloc[position] = np.nan
            for name in ("cagr", "calmar_ratio", "log_max_drawdown", "max_drawdown"):
                with self.subTest(name=name, position=position), self.assertRaisesRegex(ValueError, rf"^{name}\(\) close has 1 missing value"):
                    getattr(pandas_ta.utils, name)(gappy)

    def test_metrics_reject_invalid_numbers(self):
        # period -5, 0 or NaN read NaN and True counted as 1; benchmark_rate NaN or inf read
        # NaN or -inf (downside_deviation 0.0 for NaN); capital NaN, 0 or negative passed.
        returns = self.close.pct_change()
        cases = {
            "optimal_leverage": (self.close, {"period": (-5, 0, np.nan, True, "x"), "benchmark_rate": (np.nan, np.inf, "x"), "capital": (0, -1, np.nan)}),
            "sharpe_ratio": (self.close, {"period": (-5, 0, np.nan, True, "x"), "benchmark_rate": (np.nan, np.inf, "x")}),
            "sortino_ratio": (self.close, {"benchmark_rate": (np.nan, np.inf, "x")}),
            # (1 + rate) ** (1 / n) has no real value for a rate at or below -1
            "downside_deviation": (returns, {"benchmark_rate": (np.nan, np.inf, "x", -1, -2)}),
        }  # fmt: skip
        for name, (series, params) in cases.items():
            for param, values in params.items():
                for value in values:
                    with self.subTest(name=name, param=param, value=value), self.assertRaisesRegex(ValueError, rf"^{name}\(\) {param} "):
                        getattr(pandas_ta.utils, name)(series, **{param: value})

    def test_metrics_accept_valid_numbers(self):
        # A negative benchmark rate is a real rate; None selects the default.
        returns = self.close.pct_change()
        self.assertEqual(pandas_ta.utils.sharpe_ratio(self.close, period=None), pandas_ta.utils.sharpe_ratio(self.close))
        self.assertEqual(pandas_ta.utils.optimal_leverage(self.close, capital=None), pandas_ta.utils.optimal_leverage(self.close))
        for name, series in (("sharpe_ratio", self.close), ("sortino_ratio", self.close), ("optimal_leverage", self.close), ("downside_deviation", returns)):
            with self.subTest(name=name):
                self.assertTrue(np.isfinite(getattr(pandas_ta.utils, name)(series, benchmark_rate=-0.02)))

    def test_metrics_need_a_time_span(self):
        # They divide by total_time(): a span of 0 raised a bare ZeroDivisionError,
        # and a descending index a NaN or a negative growth rate without an error.
        one_day = self.close.iloc[:1]
        same_day = Series([0.01, -0.02], index=[self.close.index[0]] * 2)
        cases = [
            ("cagr", lambda s: pandas_ta.cagr(s), one_day),
            ("downside_deviation", lambda s: pandas_ta.downside_deviation(s), same_day),
            ("volatility", lambda s: pandas_ta.utils.volatility(s, returns=True), same_day),
        ]
        for name, func, flat in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, rf"{name}\(\) needs at least two distinct timestamps; the index spans 0 years"):
                    func(flat)
                with self.assertRaisesRegex(ValueError, r"total_time\(\) needs an index sorted in ascending order"):
                    func(self.pctret.iloc[1:][::-1] if name != "cagr" else self.close.iloc[::-1])

    def test_volatility_is_only_reachable_through_utils(self):
        """The volatility category subpackage shadows the metric of that name.

        __all__ used to list 'volatility', promising ta.volatility(close) —
        which is the subpackage and raises TypeError. Every sibling metric is
        reachable as ta.<name>; this one is the documented exception.
        """
        siblings = [
            "cagr",
            "calmar_ratio",
            "downside_deviation",
            "jensens_alpha",
            "log_max_drawdown",
            "max_drawdown",
            "optimal_leverage",
            "pure_profit_score",
            "sharpe_ratio",
            "sortino_ratio",
        ]
        for name in siblings:
            with self.subTest(name=name):
                self.assertIn(name, pandas_ta.__all__)
                self.assertTrue(callable(getattr(pandas_ta, name)))

        self.assertNotIn("volatility", pandas_ta.__all__)
        self.assertFalse(callable(pandas_ta.volatility))
        self.assertTrue(callable(pandas_ta.utils.volatility))

    def test_pure_profit_score_is_zero_without_a_correlation(self):
        # A flat series has zero variance, so linear_regression gives r = NaN
        # and the score falls back to 0 instead of NaN * cagr.
        idx = bdate_range("2021-01-04", periods=60)
        flat = Series([100.0] * 60, index=idx)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # np.corrcoef used to warn twice here
            self.assertEqual(pandas_ta.pure_profit_score(flat), 0)

    def test_pure_profit_score_rejects_missing_and_short_input(self):
        # One NaN made r NaN, so a rising series scored 0 without a word.
        idx = bdate_range("2021-01-04", periods=60)
        rising = Series(np.linspace(100.0, 160.0, 60), index=idx)
        rising.iloc[30] = np.nan
        with self.assertRaisesRegex(ValueError, r"pure_profit_score\(\) close has 1 missing value"):
            pandas_ta.pure_profit_score(rising)
        with self.assertRaisesRegex(ValueError, r"pure_profit_score\(\) needs at least 3 bars, got 2"):
            pandas_ta.pure_profit_score(Series([100.0, 101.0], index=idx[:2]))
