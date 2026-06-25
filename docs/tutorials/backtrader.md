# Integrating pandas-ta-classic with backtrader

`backtrader` is an event-driven backtesting framework. It processes data bar-by-bar through a `bt.Strategy` and expects indicators to be registered as `bt.Indicator` subclasses with named `lines`. Since `pandas-ta-classic` is vectorized (operates on full Series), the two libraries integrate through a **precompute-then-feed** pattern rather than a per-bar bridge.

---

## 1. The Integration Pattern

Backtrader allows extending `bt.feeds.PandasData` with extra columns. The approach:

1. Compute indicators over the full DataFrame with pandas-ta-classic.
2. Store results as extra columns on the DataFrame.
3. Declare those columns as `lines` on a custom `PandasData` subclass.
4. Access them in your strategy as `self.data.<column_name>`.

```python
import pandas as pd
import pandas_ta_classic as ta
import backtrader as bt

# Step 1 — precompute
df['sma_fast'] = ta.sma(df['Close'], length=10)
df['sma_slow'] = ta.sma(df['Close'], length=20)
df = df.dropna()

# Step 2 — declare extra lines
class PandasDataWithTA(bt.feeds.PandasData):
    lines = ('sma_fast', 'sma_slow',)
    params = (('sma_fast', -1), ('sma_slow', -1),)
    # param value -1 tells backtrader to auto-detect the column by line name

# Step 3 — feed to cerebro
data = PandasDataWithTA(dataname=df)
```

> **Why precompute?** Backtrader's `bt.Indicator` system calls `next()` once per bar. pandas-ta-classic indicators are designed for full-array computation, not incremental updates. Precomputing outside cerebro avoids per-bar overhead and keeps the integration clean.

---

## 2. Common Integration Patterns

### Pattern A: Trend-Following (Single Output)

Single-output indicators (SMA, RSI, EMA) produce one numpy array per indicator. Declare one line per indicator.

```python
class PandasDataWithTA(bt.feeds.PandasData):
    lines = ('sma_fast', 'sma_slow',)
    params = (('sma_fast', -1), ('sma_slow', -1),)


class SMACrossover(bt.Strategy):
    def __init__(self):
        self.crossover = bt.indicators.CrossOver(self.data.sma_fast, self.data.sma_slow)
        self.order = None

    def next(self):
        if self.order:
            return
        if self.crossover > 0 and not self.position:
            self.order = self.buy()
        elif self.crossover < 0 and self.position:
            self.order = self.sell()

    def notify_order(self, order):
        if order.status in (order.Completed, order.Cancelled, order.Rejected):
            self.order = None
```

### Pattern B: Multi-Output Indicators (MACD, Bollinger Bands)

Multi-output indicators return a DataFrame. Flatten each column into a separate line.

```python
macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
# columns: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
df['macd_line'] = macd.iloc[:, 0]
df['macd_hist'] = macd.iloc[:, 1]
df['macd_signal'] = macd.iloc[:, 2]
df = df.dropna()


class PandasDataWithTA(bt.feeds.PandasData):
    lines = ('macd_line', 'macd_hist', 'macd_signal',)
    params = (('macd_line', -1), ('macd_hist', -1), ('macd_signal', -1),)


class MACDMomentum(bt.Strategy):
    def __init__(self):
        self.crossover = bt.indicators.CrossOver(self.data.macd_line, self.data.macd_signal)
        self.order = None

    def next(self):
        if self.order:
            return
        if self.crossover > 0 and not self.position:
            self.order = self.buy()
        elif self.crossover < 0 and self.position:
            self.order = self.sell()

    def notify_order(self, order):
        if order.status in (order.Completed, order.Cancelled, order.Rejected):
            self.order = None
```

> **Column order:** `ta.macd()` returns `[MACD_line, MACD_histogram, MACD_signal]`. Use `.iloc[:, n]` by position or index by column name to be explicit.

### Pattern C: Volatility / OHLCV Indicators (ATR, ADX)

Indicators that need OHLCV data (ATR, ADX, Stochastic) work the same way — pandas-ta-classic receives the full DataFrame before cerebro starts, so there is no missing-column problem.

```python
df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
df['sma'] = ta.sma(df['Close'], length=20)
df = df.dropna()


class PandasDataWithTA(bt.feeds.PandasData):
    lines = ('atr', 'sma',)
    params = (('atr', -1), ('sma', -1),)


class ATRTrailingStop(bt.Strategy):
    def __init__(self):
        self.order = None

    def next(self):
        if self.order:
            return
        above_sma = self.data.Close[0] > self.data.sma[0]
        high_vol = self.data.atr[0] > 2.5  # scale to your instrument's price range

        if above_sma and not self.position:
            self.order = self.buy()
        elif high_vol and self.position:
            self.order = self.sell()

    def notify_order(self, order):
        if order.status in (order.Completed, order.Cancelled, order.Rejected):
            self.order = None
```

---

## 3. Dynamic Feed Helper

When strategies use many indicators, a helper that builds the `PandasData` subclass dynamically avoids repetitive boilerplate:

```python
def make_feed(df: pd.DataFrame, *extra_cols: str) -> type:
    lines = tuple(extra_cols)
    params = tuple((col, -1) for col in extra_cols)
    return type('PandasDataWithTA', (bt.feeds.PandasData,), {'lines': lines, 'params': params})

# Usage
FeedCls = make_feed(df, 'sma_fast', 'sma_slow', 'atr')
data = FeedCls(dataname=df)
```

---

## 4. Running a Backtest

```python
# SMACrossover defined in Pattern A; data built with make_feed() above
cerebro = bt.Cerebro()
cerebro.adddata(data)
cerebro.addstrategy(SMACrossover)
cerebro.broker.setcash(10_000.0)
cerebro.broker.setcommission(commission=0.002)

cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

results = cerebro.run()
strat = results[0]

print(f"Final value:   ${cerebro.broker.getvalue():,.2f}")
print(f"Sharpe ratio:  {strat.analyzers.sharpe.get_analysis().get('sharperatio', float('nan')):.2f}")
print(f"Max drawdown:  {strat.analyzers.drawdown.get_analysis().max.drawdown:.2f}%")
print(f"Total return:  {strat.analyzers.returns.get_analysis()['rtot'] * 100:.2f}%")
```

For a fully runnable script, refer to `examples/backtrader_strategy.py`.
