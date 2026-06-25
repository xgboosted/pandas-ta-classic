
# Integrating pandas-ta-classic with backtesting.py

`backtesting.py` is a powerful, event-driven backtesting library. However, it expects indicators to return pure `numpy` arrays that match the exact length of the price data. Because `pandas-ta-classic` is designed around Pandas Series and DataFrames, passing data directly between the two can cause shape errors or missing column (OHLCV) errors.

This tutorial demonstrates how to use a simple "bridge" function to securely route data between the two libraries, covering single-line indicators, multi-line indicators, and volume/volatility indicators.

---

## 1. The Bridge Function

To connect the libraries, we use a helper function inside the `self.I()` wrapper. This function does three things:
1. Reconstructs `backtesting.py`'s internal data arrays into a proper Pandas OHLCV DataFrame.
2. Validates the indicator string to prevent arbitrary code execution (reflection).
3. Detects if the indicator returns multiple columns (like MACD) and unpacks them into a tuple of numpy arrays.

```python
import pandas as pd
import pandas_ta_classic as ta

def ta_bridge(data, indicator_name, **kwargs):
    """Securely routes OHLCV data from backtesting.py to pandas-ta-classic."""
    
    # Reconstruct the DataFrame
    df = pd.DataFrame({
        'Open': data.Open,
        'High': data.High,
        'Low': data.Low,
        'Close': data.Close,
        'Volume': data.Volume
    })
    
    # Validate the indicator exists in the library
    if indicator_name not in dir(df.ta):
        raise ValueError(f"Indicator '{indicator_name}' not found.")
    
    # Execute the indicator
    indicator_method = getattr(df.ta, indicator_name)
    result = indicator_method(**kwargs)
    
    # Handle multi-output indicators (Returns a tuple of arrays)
    if isinstance(result, pd.DataFrame):
        return tuple(result[col].to_numpy() for col in result.columns)
            
    # Handle single-output indicators (Returns a single array)
    return result.to_numpy()

```

---

## 2. Common Integration Patterns

Below are examples of how to implement standard quantitative patterns inside a `backtesting.py` Strategy class using the `ta_bridge`.

### Pattern A: Trend-Following (Single Output)

For standard trend indicators like Simple Moving Averages (SMA) or Relative Strength Index (RSI), `ta_bridge` returns a single numpy array.

```python
from backtesting import Strategy
from backtesting.lib import crossover

class SMACrossover(Strategy):
    def init(self):
        # We pass the string 'sma' and the required pandas-ta kwargs
        self.sma_fast = self.I(ta_bridge, self.data, 'sma', length=10)
        self.sma_slow = self.I(ta_bridge, self.data, 'sma', length=50)

    def next(self):
        if crossover(self.sma_fast, self.sma_slow):
            self.buy()
        elif crossover(self.sma_slow, self.sma_fast):
            self.position.close()

```

### Pattern B: Momentum (Multi-Indicator Unpacking)

Indicators like MACD or Bollinger Bands return multiple lines (e.g., the MACD line, the Signal line, and the Histogram). Because `ta_bridge` converts DataFrames into tuples, you can unpack them directly into separate variables in your `init` block.

```python
class MACDMomentum(Strategy):
    def init(self):
        # Unpack the 3 distinct arrays returned by MACD
        self.macd_line, self.macd_hist, self.macd_signal = self.I(
            ta_bridge, self.data, 'macd', fast=12, slow=26, signal=9
        )

    def next(self):
        # Trigger buy when MACD line crosses above the Signal line
        if crossover(self.macd_line, self.macd_signal):
            self.buy()
        elif crossover(self.macd_signal, self.macd_line):
            self.position.close()

```

### Pattern C: Volatility (OHLCV Dependency)

Indicators like the Average True Range (ATR) or On-Balance Volume (OBV) require High, Low, Close, or Volume data to calculate. Because our `ta_bridge` reconstructs the entire OHLCV frame, you call them exactly the same way as an SMA.

```python
class ATRTrailingStop(Strategy):
    def init(self):
        # The bridge automatically passes High/Low/Close to the ATR calculation
        self.atr = self.I(ta_bridge, self.data, 'atr', length=14)
        
        # You can layer multiple pandas-ta indicators easily
        self.sma = self.I(ta_bridge, self.data, 'sma', length=20)

    def next(self):
        # Example logic using both SMA and ATR
        if self.data.Close[-1] > self.sma[-1]:
            if not self.position:
                self.buy()
                
        # Access the current ATR value for dynamic stop losses or filters
        current_volatility = self.atr[-1]
        if current_volatility > 2.5:
            # Execute high-volatility logic here
            pass

```

For a fully runnable script utilizing these patterns, refer to `examples/backtesting_py_strategy.py`.
