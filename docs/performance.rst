Performance Metrics
==================

**BETA** - Performance Metrics are a **new** addition to the package and are likely unreliable. **Use at your own risk.** These metrics return a *float* and are *not* part of the *DataFrame* Extension. They are called the Standard way.

.. code-block:: python

    import pandas_ta_classic as ta
    result = ta.cagr(df.close)

Available Metrics
-----------------

* *Compounded Annual Growth Rate*: **cagr**
* *Calmar Ratio*: **calmar_ratio**
* *Downside Deviation*: **downside_deviation**
* *Jensen's Alpha*: **jensens_alpha**
* *Log Max Drawdown*: **log_max_drawdown**
* *Max Drawdown*: **max_drawdown**
* *Pure Profit Score*: **pure_profit_score**
* *Sharpe Ratio*: **sharpe_ratio**
* *Sortino Ratio*: **sortino_ratio**
* *Volatility*: **volatility**

Backtesting with vectorbt
-------------------------

For **easier** integration with **vectorbt**'s Portfolio ``from_signals`` method, the ``ta.trend_return`` method has been replaced with ``ta.tsignals`` method to simplify the generation of trading signals.

For a comprehensive example, see the example Jupyter Notebook `VectorBT Backtest with Pandas TA <https://github.com/xgboosted/pandas-ta-classic/blob/main/examples/VectorBT_Backtest_with_Pandas_TA.ipynb>`_ in the examples directory.

Brief Example
~~~~~~~~~~~~~

.. code-block:: python

    import pandas as pd
    import pandas_ta_classic as ta
    import vectorbt as vbt

    # Load your data
    df = pd.read_csv("your_data.csv")
    
    # Generate trading signals using Pandas TA Classic
    df['sma_short'] = df.ta.sma(length=20)
    df['sma_long'] = df.ta.sma(length=50)
    
    # Create buy/sell signals
    entries = df['sma_short'] > df['sma_long']
    exits = df['sma_short'] < df['sma_long']
    
    # Use tsignals for vectorbt integration
    signals = ta.tsignals(entries, asbool=True)
    
    # Run backtest with vectorbt
    portfolio = vbt.Portfolio.from_signals(
        df['close'], 
        entries=signals['TS_Entries'], 
        exits=signals['TS_Exits']
    )
    
    # View results
    portfolio.stats()