import logging
from pandas import DataFrame
from pandas_ta_classic import Imports, version

# from .._core import _camelCase2Title
# from .._time import ytd_df

logger = logging.getLogger(__name__)


def _normalize_alpha_vantage_df(df: DataFrame, ticker: str) -> DataFrame:
    """Normalize alpha_vantage dataframe schema to OHLCV columns."""
    if df is None or df.empty:
        return DataFrame()

    rename_map = {
        "1. open": "Open",
        "2. high": "High",
        "3. low": "Low",
        "4. close": "Close",
        "5. volume": "Volume",
    }
    df = df.rename(columns=rename_map)
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    df.name = ticker
    return df


def av(ticker: str, **kwargs):
    logger.debug(f"kwargs: {kwargs}")
    verbose = kwargs.pop("verbose", False)
    kind = kwargs.pop("kind", "history")
    kind = kind.lower()
    interval = kwargs.pop("interval", "D")
    show = kwargs.pop("show", None)
    # last = kwargs.pop("last", RATE["TRADING_DAYS_PER_YEAR"])

    ticker = ticker.upper() if ticker is not None and isinstance(ticker, str) else None

    if Imports["alpha-vantage"] and ticker is not None:
        AVC = {
            "api_key": "YOUR API KEY",
            "output_size": "full",
        }
        _config = kwargs.pop("av_kwargs", AVC)
        period = kwargs.pop("period", _config.get("output_size", "full"))

        _all = ["all"]

        if kind in [*_all, "history", "h"]:
            try:
                from alpha_vantage.timeseries import TimeSeries

                ts = TimeSeries(
                    key=_config.get("api_key", "YOUR API KEY"),
                    output_format="pandas",
                )

                interval_text = str(interval).strip()
                interval_upper = interval_text.upper()
                interval_lower = interval_text.lower()

                if interval_upper in {"D", "1D", "DAILY"}:
                    df, _ = ts.get_daily(symbol=ticker, outputsize=period)
                elif interval_upper in {"W", "1W", "WEEKLY"}:
                    df, _ = ts.get_weekly(symbol=ticker)
                elif interval_upper in {"M", "MONTHLY"}:
                    df, _ = ts.get_monthly(symbol=ticker)
                else:
                    intraday_map = {
                        "1m": "1min",
                        "5m": "5min",
                        "15m": "15min",
                        "30m": "30min",
                        "60m": "60min",
                    }
                    av_interval = intraday_map.get(interval_lower, interval_lower)
                    if av_interval not in {"1min", "5min", "15min", "30min", "60min"}:
                        av_interval = "60min"
                    df, _ = ts.get_intraday(
                        symbol=ticker,
                        interval=av_interval,
                        outputsize=period,
                    )

                df = _normalize_alpha_vantage_df(df, ticker)
                if verbose:
                    logger.info("Chart History: Pandas TA v%s & alpha-vantage", version)
                    logger.info(
                        "Downloading %s[%s:%s] from Alpha Vantage",
                        ticker,
                        interval,
                        period,
                    )
                if show is not None and isinstance(show, int) and show > 0:
                    logger.info(f"\n{df.name}\n{df.tail(show)}\n")
                return df
            except ImportError:
                return DataFrame()
            except Exception:
                logger.exception("Alpha Vantage request failed for %s", ticker)
                return DataFrame()

    return DataFrame()
