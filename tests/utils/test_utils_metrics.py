import math
import warnings

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays
from pandas import DataFrame, Series, bdate_range

import pandas_ta_classic as pandas_ta
from tests.config import get_sample_data


class TestUtilityMetrics:
    @classmethod
    def setup_class(cls):
        cls.data = get_sample_data()
        cls.close = cls.data["close"]
        cls.pctret = pandas_ta.percent_return(cls.close, cumulative=False)
        cls.logret = pandas_ta.percent_return(cls.close, cumulative=False)

    @classmethod
    def teardown_class(cls):
        del cls.data
        del cls.pctret
        del cls.logret

    def setup_method(self):
        pass

    def teardown_method(self):
        pass

    def test_cagr(self):
        result = pandas_ta.utils.cagr(self.data.close)
        assert isinstance(result, float)
        assert result > 0

        # Round trip: a price series that doubles over exactly one calendar
        # year must give a CAGR of ~100%.  The old calendar-days/252 bug made
        # this ~61%.
        idx = bdate_range("2021-01-04", "2022-01-04")
        close = Series(100 * 2.0 ** ((idx - idx[0]).days / 365.0), index=idx)
        assert round(abs(pandas_ta.utils.cagr(close) - 1.0), 2) == 0

    def test_calmar_ratio(self, subtests):
        result = pandas_ta.calmar_ratio(self.close)
        assert isinstance(result, float)
        assert result >= 0

        for bad_years in (0, -2):
            with subtests.test(years=bad_years), pytest.raises(ValueError, match="years must be an integer"):
                pandas_ta.calmar_ratio(self.close, years=bad_years)

        with pytest.raises(ValueError, match=r"calmar_ratio\(\) method must be one of"):
            pandas_ta.calmar_ratio(self.close, method="bogus")

    def test_downside_deviation(self):
        result = pandas_ta.downside_deviation(self.pctret)
        assert isinstance(result, float)
        assert result >= 0

        result = pandas_ta.downside_deviation(self.logret)
        assert isinstance(result, float)
        assert result >= 0

    def test_downside_deviation_counts_only_the_returns_it_sums(self):
        # The sum skipped NaN while the divisor counted it: the leading NaN of
        # pct_change gave sqrt(ss / n), a clean series sqrt(ss / (n - 1)), and
        # a gap inside a smaller value with no error.
        returns = self.pctret
        assert np.isnan(returns.iloc[0])
        assert pandas_ta.downside_deviation(returns) == pandas_ta.downside_deviation(returns.iloc[1:])

        gappy = returns.copy()
        gappy.iloc[100] = np.nan
        with pytest.raises(ValueError, match=r"downside_deviation\(\) returns has 1 missing value"):
            pandas_ta.downside_deviation(gappy)
        with pytest.raises(ValueError, match=r"downside_deviation\(\) needs at least 2 returns, got 1"):
            pandas_ta.downside_deviation(returns.iloc[:2])

    def test_sortino_ratio_rejects_missing_close(self):
        gappy = self.close.copy()
        gappy.iloc[100] = np.nan
        with pytest.raises(ValueError, match=r"sortino_ratio\(\) close has 1 missing value"):
            pandas_ta.sortino_ratio(gappy)

    def test_return_metrics_reject_missing_close(self, subtests):
        # A gap dropped two returns from the mean and the deviation without a
        # word: sharpe_ratio read 0.3305 instead of 0.3293 for one missing close.
        gappy = self.close.copy()
        gappy.iloc[2000] = np.nan
        for name in ("sharpe_ratio", "optimal_leverage", "volatility"):
            with subtests.test(name=name), pytest.raises(ValueError, match=rf"{name}\(\) close has 1 missing value"):
                getattr(pandas_ta.utils, name)(gappy)

    def test_volatility_does_not_count_the_first_bar(self):
        # The periods per year counted the first bar, which has no return, so
        # the result moved with whether the caller dropped it (0.198789 vs 0.198783).
        assert np.isnan(self.pctret.iloc[0])
        with_nan = pandas_ta.utils.volatility(self.pctret, returns=True)
        assert with_nan == pandas_ta.utils.volatility(self.pctret.iloc[1:], returns=True)
        assert with_nan == pandas_ta.utils.volatility(self.close)

    def test_drawdown(self):
        # A return series is not a price: its percent and log drawdown read NaN
        # where it is at or below zero, with one warning (#345).
        for returns in (self.pctret, self.logret):
            with pytest.warns(UserWarning, match=r"^drawdown\(\) reads NaN on \d+ bar\(s\) whose price is at or below zero"):
                result = pandas_ta.drawdown(returns)
            assert isinstance(result, DataFrame)
            assert result.name == "DD"
            assert result["DD_PCT"][returns <= 0].isna().all()

    def test_jensens_alpha(self):
        bench_return = self.pctret.sample(n=self.close.shape[0], random_state=1)
        result = pandas_ta.jensens_alpha(self.close, bench_return)
        assert isinstance(result, float)
        assert result >= 0

        # Constructed fixture: returns = 0.0005 + 1.2 * bench.  The benchmark
        # sums to less than 1 in absolute value, which the old ``int(x.sum())
        # != 0`` guard skipped, returning NaN instead of the intercept 0.0005.
        bench = Series(np.linspace(-0.002, 0.002, 200))
        returns = 0.0005 + 1.2 * bench
        assert round(abs(pandas_ta.jensens_alpha(returns, bench) - 0.0005), 6) == 0

    def test_log_max_drawdown(self):
        result = pandas_ta.log_max_drawdown(self.close)
        assert isinstance(result, float)
        assert result >= 0

    def test_max_drawdown(self):
        result = pandas_ta.max_drawdown(self.close)
        assert isinstance(result, float)
        assert result >= 0

        result = pandas_ta.max_drawdown(self.close, method="percent")
        assert isinstance(result, float)
        assert result >= 0

        result = pandas_ta.max_drawdown(self.close, method="log")
        assert isinstance(result, float)
        assert result >= 0

        result = pandas_ta.max_drawdown(self.close, all_methods=True)
        assert isinstance(result, dict)
        assert isinstance(result["dollar"], float)
        assert isinstance(result["percent"], float)
        assert isinstance(result["log"], float)

        with pytest.raises(ValueError, match=r"method must be one of"):
            pandas_ta.max_drawdown(self.close, method="bogus")

        # the 'all' alias of all_methods was removed in 0.9.0
        with pytest.raises(TypeError, match="unexpected keyword argument 'all'"):
            pandas_ta.max_drawdown(self.close, all=True)

    def test_optimal_leverage(self):
        result = pandas_ta.optimal_leverage(self.close)
        assert isinstance(result, float)
        assert math.isfinite(result)
        result = pandas_ta.optimal_leverage(self.close, log=True)
        assert isinstance(result, float)
        assert math.isfinite(result)
        constant = Series([100.0] * 60)
        with pytest.raises(ValueError):
            pandas_ta.optimal_leverage(constant)

    def test_pure_profit_score(self):
        result = pandas_ta.pure_profit_score(self.close)
        assert result >= 0

        # A strictly linear series correlates r = 1.0 with its time index, so
        # the score equals the CAGR.  The old constant-zeros time index made the
        # correlation NaN and the function always returned 0.
        idx = bdate_range("2021-01-04", periods=250)
        linear = Series(100.0 + np.arange(250), index=idx)
        assert round(abs(pandas_ta.pure_profit_score(linear) - pandas_ta.cagr(linear)), 8) == 0

    def test_sharpe_ratio(self):
        result = pandas_ta.sharpe_ratio(self.close)
        assert isinstance(result, float)
        assert result >= 0

        result_cagr = pandas_ta.sharpe_ratio(self.close, use_cagr=True)
        assert isinstance(result_cagr, float)
        assert result_cagr >= 0
        assert math.isfinite(result_cagr)

        result_log = pandas_ta.sharpe_ratio(self.close, use_cagr=True, log=True)
        assert isinstance(result_log, float)
        assert result_log >= 0
        assert math.isfinite(result_log)

    def test_sharpe_ratio_use_cagr_subtracts_benchmark_rate(self, subtests):
        # use_cagr=True divides cagr - benchmark_rate by the volatility; it used
        # to drop the rate, so every benchmark_rate returned the rate-0 value.
        rate = 0.05
        for log in (False, True):
            with subtests.test(log=log):
                expected = (pandas_ta.utils.cagr(self.close) - rate) / pandas_ta.utils.volatility(self.close, log=log)
                result = pandas_ta.sharpe_ratio(self.close, benchmark_rate=rate, use_cagr=True, log=log)
                assert result == pytest.approx(expected)
                assert result < pandas_ta.sharpe_ratio(self.close, use_cagr=True, log=log)

    def test_sortino_ratio(self):
        result = pandas_ta.sortino_ratio(self.close)
        assert isinstance(result, float)
        assert result >= 0

    def test_sortino_ratio_uses_benchmark_rate_in_the_denominator(self):
        # The denominator must measure downside below the same target the
        # numerator subtracts; it used to always measure below 0.0.
        rate = 0.05
        numerator = pandas_ta.utils.cagr(self.close) - rate
        denominator = pandas_ta.downside_deviation(self.pctret, benchmark_rate=rate)
        assert pandas_ta.sortino_ratio(self.close, benchmark_rate=rate) == pytest.approx(numerator / denominator)

    def test_volatility(self, subtests):
        returns_ = pandas_ta.percent_return(self.close)
        result = pandas_ta.utils.volatility(returns_, returns=True)
        assert isinstance(result, float)
        assert result >= 0

        # Annualised daily volatility must land near stddev * sqrt(252).  The old
        # calendar-days/252 bug understated it by ~17% (sqrt(173.6) vs sqrt(252)).
        expected = float(returns_.std() * np.sqrt(252))
        assert abs(result - expected) <= (expected * 0.02)

        for tf in ["years", "months", "weeks", "days", "hours", "minutes", "seconds"]:
            result = pandas_ta.utils.volatility(self.close, tf)
            with subtests.test(tf=tf):
                assert isinstance(result, float)
                assert result >= 0

        # nearest_day rounds the bars-per-year factor up, so it cannot lower the
        # annualised figure.
        yearly = pandas_ta.utils.volatility(self.close, "years")
        nearest = pandas_ta.utils.volatility(self.close, "years", nearest_day=True)
        assert nearest >= yearly

    def test_volatility_names_a_missing_close(self):
        # The other metrics are in test_every_utility_names_a_required_input_passed_as_none;
        # volatility is kept out of utils.__all__, so that sweep does not see it. All of them
        # answered None with NaN, a sentinel AGENTS.md rule 1 rules out for utils/_metrics.py.
        with pytest.raises(ValueError, match=r"^volatility\(\) requires 'close', got None$"):
            pandas_ta.utils.volatility(None)

    def test_close_metrics_reject_missing_close(self, subtests):
        # A NaN on the first or last bar made cagr and log_max_drawdown read NaN (and
        # calmar_ratio for the last bar); one inside, or max_drawdown's anywhere, was
        # skipped without a word. The other close metrics already raised for any NaN.
        for position in (0, 2000, -1):
            gappy = self.close.copy()
            gappy.iloc[position] = np.nan
            for name in ("cagr", "calmar_ratio", "log_max_drawdown", "max_drawdown"):
                with subtests.test(name=name, position=position), pytest.raises(ValueError, match=rf"^{name}\(\) close has 1 missing value"):
                    getattr(pandas_ta.utils, name)(gappy)

    def test_metrics_reject_invalid_numbers(self, subtests):
        # period -5, 0 or NaN read NaN and True counted as 1; benchmark_rate NaN or inf read
        # NaN or -inf (downside_deviation 0.0 for NaN); capital NaN, 0 or negative passed.
        returns = self.close.pct_change()
        cases = {
            "optimal_leverage": (self.close, {"period": (-5, 0, np.nan, True, "x"), "benchmark_rate": (np.nan, np.inf, "x"), "capital": (0, -1, np.nan)}),
            "sharpe_ratio": (self.close, {"period": (-5, 0, np.nan, True, "x"), "benchmark_rate": (np.nan, np.inf, "x")}),
            "sortino_ratio": (self.close, {"benchmark_rate": (np.nan, np.inf, "x", -1, -2)}),
            # (1 + rate) ** (1 / n) has no real value for a rate at or below -1
            "downside_deviation": (returns, {"benchmark_rate": (np.nan, np.inf, "x", -1, -2)}),
        }  # fmt: skip
        for name, (series, params) in cases.items():
            for param, values in params.items():
                for value in values:
                    with subtests.test(name=name, param=param, value=value), pytest.raises(ValueError, match=rf"^{name}\(\) {param} "):
                        getattr(pandas_ta.utils, name)(series, **{param: value})

    def test_metrics_accept_valid_numbers(self, subtests):
        # A negative benchmark rate is a real rate; None selects the default.
        returns = self.close.pct_change()
        assert pandas_ta.utils.sharpe_ratio(self.close, period=None) == pandas_ta.utils.sharpe_ratio(self.close)
        assert pandas_ta.utils.optimal_leverage(self.close, capital=None) == pandas_ta.utils.optimal_leverage(self.close)
        for name, series in (("sharpe_ratio", self.close), ("sortino_ratio", self.close), ("optimal_leverage", self.close), ("downside_deviation", returns)):
            with subtests.test(name=name, benchmark_rate=None):
                func = getattr(pandas_ta.utils, name)
                assert func(series, benchmark_rate=None) == func(series)
        for name, series in (("sharpe_ratio", self.close), ("sortino_ratio", self.close), ("optimal_leverage", self.close), ("downside_deviation", returns)):
            with subtests.test(name=name):
                assert np.isfinite(getattr(pandas_ta.utils, name)(series, benchmark_rate=-0.02))

    def test_metrics_need_a_time_span(self, subtests):
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
            with subtests.test(name=name):
                with pytest.raises(ValueError, match=rf"{name}\(\) needs at least two distinct timestamps; the index spans 0 years"):
                    func(flat)
                with pytest.raises(ValueError, match=r"total_time\(\) needs an index sorted in ascending order"):
                    func(self.pctret.iloc[1:][::-1] if name != "cagr" else self.close.iloc[::-1])

    def test_volatility_is_only_reachable_through_utils(self, subtests):
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
            with subtests.test(name=name):
                assert name in pandas_ta.__all__
                assert callable(getattr(pandas_ta, name))

        assert "volatility" not in pandas_ta.__all__
        assert not callable(pandas_ta.volatility)
        assert callable(pandas_ta.utils.volatility)

    def test_pure_profit_score_is_zero_without_a_correlation(self):
        # A flat series has zero variance, so linear_regression gives r = NaN
        # and the score falls back to 0 instead of NaN * cagr.
        idx = bdate_range("2021-01-04", periods=60)
        flat = Series([100.0] * 60, index=idx)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # np.corrcoef used to warn twice here
            assert pandas_ta.pure_profit_score(flat) == 0

    def test_pure_profit_score_rejects_missing_and_short_input(self):
        # One NaN made r NaN, so a rising series scored 0 without a word.
        idx = bdate_range("2021-01-04", periods=60)
        rising = Series(np.linspace(100.0, 160.0, 60), index=idx)
        rising.iloc[30] = np.nan
        with pytest.raises(ValueError, match=r"pure_profit_score\(\) close has 1 missing value"):
            pandas_ta.pure_profit_score(rising)
        with pytest.raises(ValueError, match=r"pure_profit_score\(\) needs at least 3 bars, got 2"):
            pandas_ta.pure_profit_score(Series([100.0, 101.0], index=idx[:2]))


_returns = arrays(np.float64, st.integers(min_value=2, max_value=200), elements=st.floats(min_value=-0.2, max_value=0.2, width=64))


def _daily(values: np.ndarray) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2021-01-04", periods=len(values)))


class TestDownsideDeviationProperties:
    """Property tests for ``downside_deviation``.

    Its sum of squares skipped NaN while the divisor counted it, so the result
    depended on whether the leading NaN of a percent return was still there,
    and a gap inside the series gave a smaller value without an error.
    """

    @given(_returns, st.floats(min_value=-0.1, max_value=0.2, width=64))
    def test_matches_the_definition(self, values, rate):
        returns = _daily(values)
        years = (returns.index[-1] - returns.index[0]).total_seconds() / (365.25 * 86400)
        periods = values.size / years
        target = (1 + rate) ** (1 / periods) - 1
        expected = math.sqrt(np.sum(np.minimum(values - target, 0.0) ** 2) / values.size) * math.sqrt(periods)
        assert pandas_ta.downside_deviation(returns, benchmark_rate=rate) == pytest.approx(expected, rel=1e-9, abs=1e-12)

    @given(_returns, st.integers(min_value=0, max_value=5), st.integers(min_value=0, max_value=5))
    def test_leading_and_trailing_nan_are_not_counted(self, values, lead, trail):
        padded = _daily(np.concatenate([np.full(lead, np.nan), values, np.full(trail, np.nan)]))
        trimmed = padded.iloc[lead : lead + values.size]
        assert pandas_ta.downside_deviation(padded) == pandas_ta.downside_deviation(trimmed)

    @given(arrays(np.float64, st.integers(min_value=3, max_value=200), elements=st.floats(min_value=-0.2, max_value=0.2, width=64)), st.data())
    def test_nan_inside_raises(self, values, data):
        pos = data.draw(st.integers(min_value=1, max_value=values.size - 2))
        values[pos] = np.nan
        with pytest.raises(ValueError, match="returns has 1 missing value"):
            pandas_ta.downside_deviation(_daily(values))


# Every metric that turns a close into returns: their means and deviations skip
# NaN, so a missing close silently dropped two returns (sharpe 0.3293 -> 0.3305).


_METRICS_ON_CLOSE = {
    "optimal_leverage": lambda s: pandas_ta.utils.optimal_leverage(s),
    "pure_profit_score": lambda s: pandas_ta.utils.pure_profit_score(s),
    "sharpe_ratio": lambda s: pandas_ta.utils.sharpe_ratio(s),
    "sharpe_ratio(use_cagr)": lambda s: pandas_ta.utils.sharpe_ratio(s, use_cagr=True),
    "sortino_ratio": lambda s: pandas_ta.utils.sortino_ratio(s),
    "volatility": lambda s: pandas_ta.utils.volatility(s),
    "volatility(log)": lambda s: pandas_ta.utils.volatility(s, log=True),
}


class TestMetricsOnCloseProperties:
    """Property tests for the metrics computed from a close's returns."""

    @given(
        arrays(np.float64, st.integers(min_value=5, max_value=200), elements=st.floats(min_value=1.0, max_value=1000.0, width=64)),
        st.sampled_from(sorted(_METRICS_ON_CLOSE)),
        st.data(),
    )
    def test_a_missing_close_anywhere_raises(self, values, name, data):
        values[data.draw(st.integers(min_value=0, max_value=values.size - 1))] = np.nan
        with pytest.raises(ValueError, match=rf"^{name.split('(')[0]}\(\) close has 1 missing value"):
            _METRICS_ON_CLOSE[name](_daily(values))

    @given(_returns, st.integers(min_value=0, max_value=5), st.integers(min_value=0, max_value=5))
    def test_volatility_does_not_count_leading_and_trailing_nan(self, values, lead, trail):
        # The periods per year counted the leading NaN of a percent return that
        # the deviation skipped: the result depended on whether it was passed.
        padded = _daily(np.concatenate([np.full(lead, np.nan), values, np.full(trail, np.nan)]))
        trimmed = padded.iloc[lead : lead + values.size]
        assert pandas_ta.utils.volatility(padded, returns=True) == pandas_ta.utils.volatility(trimmed, returns=True)

    @given(arrays(np.float64, st.integers(min_value=3, max_value=200), elements=st.floats(min_value=1.0, max_value=1000.0, width=64)), st.booleans())
    def test_volatility_of_a_close_equals_that_of_its_returns(self, values, log):
        close = _daily(values)
        returns = pandas_ta.log_return(close) if log else pandas_ta.percent_return(close)
        assert pandas_ta.utils.volatility(close, log=log) == pytest.approx(pandas_ta.utils.volatility(returns, returns=True), rel=1e-12)


class TestMetricsSpanProperties:
    """Properties of the time span the metrics divide by."""

    @given(
        st.integers(min_value=3, max_value=30),
        st.datetimes(min_value=pd.Timestamp("1990-01-01").to_pydatetime(), max_value=pd.Timestamp("2040-01-01").to_pydatetime()),
        st.sampled_from(["cagr", "downside_deviation", "volatility"]),
    )
    def test_metrics_reject_a_zero_span(self, n, stamp, name):
        """Rows sharing one timestamp span 0; dividing by it raised a bare ZeroDivisionError."""
        series = pd.Series(np.linspace(100.0, 101.0, n), index=pd.DatetimeIndex([stamp] * n))
        calls = {
            "cagr": lambda: pandas_ta.cagr(series),
            "downside_deviation": lambda: pandas_ta.downside_deviation(series.pct_change()),
            "volatility": lambda: pandas_ta.utils.volatility(series),
        }
        with pytest.raises(ValueError, match=rf"^{name}\(\) needs at least two distinct timestamps"):
            calls[name]()
