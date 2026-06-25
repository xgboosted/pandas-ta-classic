Performance Metrics
==================

Performance Metrics return a *float* and are *not* part of the *DataFrame* Extension. They are called the Standard way. The DataFrame must have a **DatetimeIndex** for time-based metrics such as ``cagr``.

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

Backtesting
-----------

**Pandas TA Classic** provides trading signals and indicator output
suitable for popular backtesting frameworks.

vectorbt
~~~~~~~~

Use ``ta.tsignals`` to generate entry/exit signals for **vectorbt**'s
``Portfolio.from_signals`` method.

For a comprehensive example, see the Jupyter Notebook `VectorBT Backtest
with Pandas TA <https://github.com/xgboosted/pandas-ta-classic/blob/main/examples/VectorBT_Backtest_with_Pandas_TA.ipynb>`_
in the examples directory.

.. code-block:: python

    import pandas as pd
    import pandas_ta_classic as ta
    import vectorbt as vbt

    df = pd.read_csv("your_data.csv")

    df['sma_short'] = df.ta.sma(length=20)
    df['sma_long'] = df.ta.sma(length=50)

    entries = df['sma_short'] > df['sma_long']
    exits = df['sma_short'] < df['sma_long']

    signals = ta.tsignals(entries, asbool=True)

    portfolio = vbt.Portfolio.from_signals(
        df['close'],
        entries=signals['TS_Entries'],
        exits=signals['TS_Exits'],
    )

    portfolio.stats()

backtesting.py
~~~~~~~~~~~~~~

For **backtesting.py**, use a bridge function to feed pandas-ta-classic
indicator output into the backtester. See the :doc:`integration tutorial
<tutorials/backtesting_py>` for a full walkthrough and the runnable
`examples/backtesting_py_strategy.py <https://github.com/xgboosted/pandas-ta-classic/blob/main/examples/backtesting_py_strategy.py>`_
script.

.. code-block:: python

    import pandas as pd
    import pandas_ta_classic as ta
    from backtesting import Backtest, Strategy
    from backtesting.test import GOOG
    from backtesting.lib import crossover


    def ta_bridge(data, indicator_fn):
        df = pd.DataFrame({
            'Open': data.Open, 'High': data.High,
            'Low': data.Low, 'Close': data.Close,
            'Volume': data.Volume,
        })
        result = indicator_fn(df)
        if isinstance(result, pd.DataFrame):
            return tuple(result[col].to_numpy() for col in result.columns)
        return result.to_numpy()


    class SMACrossover(Strategy):
        fast_length = 10
        slow_length = 20

        def init(self):
            self.sma_fast = self.I(
                ta_bridge, self.data,
                lambda df: df.ta.sma(length=self.fast_length),
            )
            self.sma_slow = self.I(
                ta_bridge, self.data,
                lambda df: df.ta.sma(length=self.slow_length),
            )

        def next(self):
            if crossover(self.sma_fast, self.sma_slow):
                self.buy()
            elif crossover(self.sma_slow, self.sma_fast):
                self.position.close()


    bt = Backtest(GOOG, SMACrossover, cash=10000, commission=0.002)
    bt.run()