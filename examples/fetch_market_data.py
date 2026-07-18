"""Fetch OHLCV market data, then run pandas-ta-classic indicators on it.

Data fetching is intentionally *out of scope* for pandas-ta-classic: the built-in
``df.ta.ticker()`` / ``ta.yf()`` / ``ta.av()`` helpers are deprecated and will be
removed in a future release. Fetch data with the provider library directly and
pass the resulting DataFrame to the library. This file shows the replacement
patterns for yfinance and Alpha Vantage.

Install only what you need:

    pip install yfinance          # Yahoo Finance
    pip install alpha-vantage     # Alpha Vantage (needs a free API key)
"""

import pandas_ta_classic as ta


def fetch_yfinance(ticker: str = "SPY", period: str = "1y", interval: str = "1d"):
    """Download OHLCV from Yahoo Finance and add a couple of indicators."""
    import yfinance as yf

    df = yf.Ticker(ticker).history(period=period, interval=interval)
    if df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker!r}")

    # pandas-ta-classic works on any OHLCV DataFrame regardless of source.
    df.ta.rsi(length=14, append=True)
    df.ta.macd(append=True)
    return df


def fetch_alpha_vantage(ticker: str = "SPY", api_key: str | None = None):
    """Download daily OHLCV from Alpha Vantage and add an indicator.

    Get a free API key at https://www.alphavantage.co/support/#api-key and pass
    it as ``api_key`` (or set it however your app manages secrets — never commit
    keys to source control).
    """
    from alpha_vantage.timeseries import TimeSeries

    if not api_key:
        raise ValueError("An Alpha Vantage API key is required.")

    ts = TimeSeries(key=api_key, output_format="pandas")
    df, _ = ts.get_daily(symbol=ticker, outputsize="full")

    # Alpha Vantage columns are like "1. open"; normalize to OHLCV.
    df = df.rename(
        columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. volume": "Volume",
        }
    ).sort_index()

    df.ta.sma(length=50, append=True)
    return df


if __name__ == "__main__":
    print(f"pandas-ta-classic v{ta.version}")
    out = fetch_yfinance("AAPL", period="6mo")
    print(out.tail())
