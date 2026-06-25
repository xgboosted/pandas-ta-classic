# Integrating pandas-ta-classic with backtesting.py

`backtesting.py` is a powerful, event-driven backtesting library. It expects indicators to return pure `numpy` arrays that match the exact length of the price data. Since `pandas-ta-classic` is designed around Pandas Series and DataFrames, passing data directly between the two can cause shape errors or missing column (OHLCV) errors.

This tutorial demonstrates how to use a simple bridge function to route data between the two libraries, covering single-line indicators, multi-line indicators, and volume/volatility indicators.

---

## 1. The Bridge Function

The bridge reconstructs `backtesting.py`'s internal arrays into a Pandas DataFrame, calls a user-supplied indicator function on it, and converts the result into numpy arrays that `backtesting.py` expects. No string-based reflection — you pass the indicator as a callable.

```python
import pandas as pd
import pandas_ta_classic as ta


def ta_bridge(data, indicator_fn):
    df = pd.DataFrame({
        'Open': data.Open,
        'High': data.High,
        'Low': data.Low,
        'Close': data.Close,
        'Volume': data.Volume,
    })
    result = indicator_fn(df)
    if isinstance(result, pd.DataFrame):
        return tuple(result[col].to_numpy() for col in result.columns)
    return result.to_numpy()
```

- **OHLCV**: The full OHLCV DataFrame is reconstructed so indicators like ATR and ADX get all the columns they need.
- **Multi-output**: If the indicator returns a DataFrame (e.g., MACD, Bollinger Bands), the bridge unpacks it into a tuple of numpy arrays.
- **Callable dispatch**: You supply the indicator as a lambda or function, avoiding unsafe string-based dispatch.

> **Performance note:** `backtesting.py` calls `ta_bridge` once per `self.I()` invocation in `init()`, over the full data array, then caches the result. The bridge does not run per bar. For strategies with many indicators, the cost is one DataFrame construction per indicator registration — not per tick.

---

## 2. Common Integration Patterns

### Pattern A: Trend-Following (Single Output)

For single-output indicators like SMA or RSI, `ta_bridge` returns a single numpy array.

```python
from backtesting import Strategy
from backtesting.lib import crossover


class SMACrossover(Strategy):
    fast_length = 10
    slow_length = 20

    def init(self):
        self.sma_fast = self.I(
            ta_bridge, self.data, lambda df: df.ta.sma(length=self.fast_length)
        )
        self.sma_slow = self.I(
            ta_bridge, self.data, lambda df: df.ta.sma(length=self.slow_length)
        )

    def next(self):
        if crossover(self.sma_fast, self.sma_slow):
            if not self.position:
                self.buy()
        elif crossover(self.sma_slow, self.sma_fast):
            if self.position:
                self.position.close()
```

### Pattern B: Momentum (Multi-Indicator Unpacking)

Indicators like MACD and Bollinger Bands return multiple columns. The bridge converts the DataFrame into a tuple, which you unpack directly in `init()`.

```python
class MACDMomentum(Strategy):
    def init(self):
        self.macd_line, self.macd_hist, self.macd_signal = self.I(
            ta_bridge, self.data, lambda df: df.ta.macd(fast=12, slow=26, signal=9)
        )

    def next(self):
        if crossover(self.macd_line, self.macd_signal):
            self.buy()
        elif crossover(self.macd_signal, self.macd_line):
            self.position.close()
```

> **Column order:** `df.ta.macd()` returns columns in the order `[MACD_line, MACD_histogram, MACD_signal]`. The unpacking above relies on this order. Verify column names with `result.columns` if the order changes in a future version.

### Pattern C: Volatility (OHLCV Dependency)

Indicators like ATR require High, Low, and Close. The bridge reconstructs the full OHLCV DataFrame, so these work without any extra setup.

```python
class ATRTrailingStop(Strategy):
    def init(self):
        self.atr = self.I(ta_bridge, self.data, lambda df: df.ta.atr(length=14))
        self.sma = self.I(ta_bridge, self.data, lambda df: df.ta.sma(length=20))

    def next(self):
        if self.data.Close[-1] > self.sma[-1]:
            if not self.position:
                self.buy()

        current_volatility = self.atr[-1]
        if current_volatility > 2.5:
            if self.position:
                self.position.close()
```

For a fully runnable script, refer to `examples/backtesting_py_strategy.py`.
