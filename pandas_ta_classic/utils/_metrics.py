from typing import Any, Optional, Union, cast

import numpy as np
from pandas import Series, Timedelta


from ._core import verify_series
from ._time import total_time
from ._math import linear_regression
from pandas_ta_classic import RATE
from pandas_ta_classic.performance.drawdown import drawdown
from pandas_ta_classic.performance.log_return import log_return
from pandas_ta_classic.performance.percent_return import percent_return


def cagr(close: Series) -> float:
    """Compounded Annual Growth Rate

    Args:
        close (pd.Series): Series of 'close's

    >>> result = ta.cagr(df.close)
    """
    close = verify_series(close)
    if close is None:
        return np.nan
    start, end = close.iloc[0], close.iloc[-1]
    return ((end / start) ** (1 / total_time(close))) - 1


def calmar_ratio(close: Series, method: str = "percent", years: int = 3) -> float:
    """The Calmar Ratio is the percent Max Drawdown Ratio 'typically' over
    the past three years.

    Args:
        close (pd.Series): Series of 'close's
        method (str): Max DD calculation options: 'dollar', 'percent', 'log'.
            Default: 'dollar'
        years (int): The positive number of years to use. Default: 3

    >>> result = ta.calmar_ratio(close, method="percent", years=3)
    """
    if years <= 0:
        # Guard: years must be positive and nonzero
        return np.nan
    close = verify_series(close)
    if close is None:
        return np.nan

    n_years_ago = close.index[-1] - Timedelta(days=365.25 * years)
    close = close[close.index > n_years_ago]

    return cagr(close) / cast(float, max_drawdown(close, method=method))


def downside_deviation(returns: Series, benchmark_rate: float = 0.0, tf: str = "years") -> float:
    """Downside Deviation for the Sortino ratio.
    Benchmark rate is assumed to be annualized. Adjusted according for the
    number of periods per year seen in the data.

    Args:
        close (pd.Series): Series of 'close's
        benchmark_rate (float): Benchmark Rate to use. Default: 0.0
        tf (str): Time Frame options: 'days', 'weeks', 'months', and 'years'.
            Default: 'years'

    >>> result = ta.downside_deviation(returns, benchmark_rate=0.0, tf="years")
    """
    # For both de-annualizing the benchmark rate and annualizing result
    returns = verify_series(returns)
    if returns is None:
        return np.nan
    days_per_year = returns.shape[0] / total_time(returns, tf)

    adjusted_benchmark_rate = ((1 + benchmark_rate) ** (1 / days_per_year)) - 1

    downside = adjusted_benchmark_rate - returns
    downside_sum_of_squares = (downside[downside > 0] ** 2).sum()
    downside_deviation = np.sqrt(downside_sum_of_squares / (returns.shape[0] - 1))
    return downside_deviation * np.sqrt(days_per_year)


def jensens_alpha(returns: Series, benchmark_returns: Series) -> float:
    """Jensen's 'Alpha' of a series and a benchmark.

    Args:
        returns (pd.Series): Series of 'returns's
        benchmark_returns (pd.Series): Series of 'benchmark_returns's

    >>> result = ta.jensens_alpha(returns, benchmark_returns)
    """
    returns = verify_series(returns)
    benchmark_returns = verify_series(benchmark_returns)
    if returns is None or benchmark_returns is None:
        return np.nan

    benchmark_returns = benchmark_returns.interpolate()
    return linear_regression(benchmark_returns, returns)["a"]


def log_max_drawdown(close: Series) -> float:
    """Log Max Drawdown of a series.

    Args:
        close (pd.Series): Series of 'close's

    >>> result = ta.log_max_drawdown(close)
    """
    close = verify_series(close)
    if close is None:
        return np.nan
    log_return = np.log(close.iloc[-1]) - np.log(close.iloc[0])
    return log_return - max_drawdown(close, method="log")


def max_drawdown(close: Series, method: Optional[str] = None, all: bool = False) -> Union[float, dict[str, float]]:
    """Maximum Drawdown from close. Default: 'dollar'.

    Args:
        close (pd.Series): Series of 'close's
        method (str): Max DD calculation options: 'dollar', 'percent', 'log'.
            Default: 'dollar'
        all (bool): If True, it returns all three methods as a dict.
            Default: False

    >>> result = ta.max_drawdown(close, method="dollar", all=False)
    """
    close = verify_series(close)
    if close is None:
        return np.nan
    max_dd = drawdown(close).max()

    max_dd_ = {
        "dollar": max_dd.iloc[0],
        "percent": max_dd.iloc[1],
        "log": max_dd.iloc[2],
    }
    if all:
        return max_dd_

    if isinstance(method, str) and method in max_dd_:
        return max_dd_[method]
    return max_dd_["dollar"]


def optimal_leverage(
    close: Series,
    benchmark_rate: float = 0.0,
    period: Union[float, int] = RATE["TRADING_DAYS_PER_YEAR"],
    log: bool = False,
    capital: float = 1.0,
) -> float:
    """Optimal Leverage (Kelly-like) of a series.

    Args:
        close (pd.Series): Series of 'close's
        benchmark_rate (float): Benchmark Rate to use. Default: 0.0
        period (int, float): Period to use to calculate Mean Annual Return and
            Annual Standard Deviation.
            Default: RATE["TRADING_DAYS_PER_YEAR"] (252)
        log (bool): If True, calculates log_return. Otherwise it returns
            percent_return. Default: False
        capital (float): Capital to scale the optimal leverage by. Default: 1.0

    >>> result = ta.optimal_leverage(close, benchmark_rate=0.0, log=False)
    """
    close = verify_series(close)
    if close is None:
        return np.nan

    returns = percent_return(close=close) if not log else log_return(close=close)

    period_mu = period * returns.mean()
    period_std = np.sqrt(period) * returns.std()
    if period_std == 0:
        raise ValueError("optimal_leverage: standard deviation of returns is zero")

    mean_excess_return = period_mu - benchmark_rate
    opt_leverage = (period_std**-2) * mean_excess_return

    return capital * opt_leverage


def pure_profit_score(close: Series) -> Union[float, int]:
    """Pure Profit Score of a series.

    Args:
        close (pd.Series): Series of 'close's

    >>> result = ta.pure_profit_score(df.close)
    """
    close = verify_series(close)
    if close is None:
        return np.nan
    close_index = Series(0, index=close.reset_index().index)

    r = linear_regression(close_index, close)["r"]
    if not np.isnan(r):
        return r * cagr(close)
    return 0


def sharpe_ratio(
    close: Series,
    benchmark_rate: float = 0.0,
    log: bool = False,
    use_cagr: bool = False,
    period: Union[float, int] = RATE["TRADING_DAYS_PER_YEAR"],
) -> float:
    """Sharpe Ratio of a series.

    Args:
        close (pd.Series): Series of 'close's
        benchmark_rate (float): Benchmark Rate to use. Default: 0.0
        log (bool): If True, calculates log_return. Otherwise it returns
            percent_return. Default: False
        use_cagr (bool): Use cagr - benchmark_rate instead. Default: False
        period (int, float): Period to use to calculate Mean Annual Return and
            Annual Standard Deviation.
            Default: RATE["TRADING_DAYS_PER_YEAR"] (currently 252)

    >>> result = ta.sharpe_ratio(close, benchmark_rate=0.0, log=False)
    """
    close = verify_series(close)
    if close is None:
        return np.nan
    returns = percent_return(close=close) if not log else log_return(close=close)

    if use_cagr:
        return cagr(close) / volatility(close, log=log)
    period_mu = period * returns.mean()
    period_std = np.sqrt(period) * returns.std()
    return (period_mu - benchmark_rate) / period_std


def sortino_ratio(close: Series, benchmark_rate: float = 0.0, log: bool = False) -> float:
    """Sortino Ratio of a series.

    Args:
        close (pd.Series): Series of 'close's
        benchmark_rate (float): Benchmark Rate to use. Default: 0.0
        log (bool): If True, calculates log_return. Otherwise it returns
            percent_return. Default: False

    >>> result = ta.sortino_ratio(close, benchmark_rate=0.0, log=False)
    """
    close = verify_series(close)
    if close is None:
        return np.nan
    returns = percent_return(close=close) if not log else log_return(close=close)

    result = cagr(close) - benchmark_rate
    result /= downside_deviation(returns)
    return result


def volatility(
    close: Series,
    tf: str = "years",
    returns: bool = False,
    log: bool = False,
    **kwargs: Any,
) -> float:
    """Volatility of a series. Default: 'years'

    Args:
        close (pd.Series): Series of 'close's
        tf (str): Time Frame options: 'days', 'weeks', 'months', and 'years'.
            Default: 'years'
        returns (bool): If True, then it replace the close Series with the user
            defined Series; typically user generated returns or percent returns
            or log returns. Default: False
        log (bool): If True, calculates log_return. Otherwise it calculates
            percent_return. Default: False

    >>> result = ta.volatility(close, tf="years", returns=False, log=False, **kwargs)
    """
    close = verify_series(close)
    if close is None:
        return np.nan

    _returns: Series
    if not returns:
        _returns = percent_return(close=close) if not log else log_return(close=close)
    else:
        _returns = close

    factor = _returns.shape[0] / total_time(_returns, tf)
    if kwargs.pop("nearest_day", False) and tf.lower() == "years":
        factor = int(factor + 1)
    return float(np.sqrt(factor) * _returns.std())
