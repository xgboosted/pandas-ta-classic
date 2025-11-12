# 🚀 Pandas TA Classic - Quickstart Guide

Welcome to **Pandas TA Classic**! This guide will help you get started quickly with calculating technical indicators for your financial data analysis.

## 📋 Table of Contents

- [Installation](#installation)
- [Your First Indicators](#your-first-indicators)
- [Working with Real Data](#working-with-real-data)
- [Common Use Cases](#common-use-cases)
- [Next Steps](#next-steps)

## 🔧 Installation

### Quick Install

The fastest way to get started:

```bash
pip install pandas-ta-classic
```

Or using `uv` (faster):

```bash
uv pip install pandas-ta-classic
```

### With Optional Dependencies

For full functionality including TA-Lib candlestick patterns:

```bash
pip install pandas-ta-classic TA-Lib
```

### Verify Installation

```python
import pandas_ta_classic as ta
print(ta.version)
```

## 🎯 Your First Indicators

### Method 1: Standard Approach (Explicit)

Perfect for when you want full control:

```python
import pandas as pd
import pandas_ta_classic as ta

# Create sample data
df = pd.DataFrame({
    'open': [100, 101, 102, 103, 104],
    'high': [105, 106, 107, 108, 109],
    'low': [99, 100, 101, 102, 103],
    'close': [104, 105, 106, 107, 108],
    'volume': [1000, 1100, 1200, 1300, 1400]
})

# Calculate a Simple Moving Average
sma_20 = ta.sma(df['close'], length=20)
print(sma_20)

# Calculate RSI (Relative Strength Index)
rsi_14 = ta.rsi(df['close'], length=14)
print(rsi_14)

# Calculate MACD
macd = ta.macd(df['close'])
print(macd)  # Returns a DataFrame with MACD, signal, and histogram
```

### Method 2: DataFrame Extension (Convenient)

The easiest way for most users:

```python
import pandas as pd
import pandas_ta_classic as ta

# Load your data
df = pd.read_csv('your_data.csv')

# Calculate and append indicators directly
df.ta.sma(length=20, append=True)         # Adds SMA_20 column
df.ta.rsi(length=14, append=True)         # Adds RSI_14 column
df.ta.macd(append=True)                   # Adds MACD columns
df.ta.bbands(length=20, append=True)      # Adds Bollinger Bands

# View your DataFrame with new indicators
print(df.tail())
```

### Method 3: Strategy System (Powerful)

Run multiple indicators at once with multiprocessing:

```python
import pandas as pd
import pandas_ta_classic as ta

df = pd.read_csv('your_data.csv')

# Use a built-in strategy
df.ta.strategy("CommonStrategy")

# Or create your own custom strategy
my_strategy = ta.Strategy(
    name="MyStrategy",
    ta=[
        {"kind": "sma", "length": 20},
        {"kind": "sma", "length": 50},
        {"kind": "rsi", "length": 14},
        {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
        {"kind": "bbands", "length": 20},
    ]
)

# Run your strategy
df.ta.strategy(my_strategy)
print(df.columns)  # See all new indicator columns
```

## 📊 Working with Real Data

### Using yfinance (Recommended)

```python
import pandas as pd
import pandas_ta_classic as ta
import yfinance as yf

# Download stock data
df = yf.download("AAPL", start="2023-01-01", end="2024-01-01")

# Calculate indicators
df.ta.sma(length=50, append=True)
df.ta.rsi(length=14, append=True)
df.ta.macd(append=True)

# View results
print(df.tail())
```

### Using Built-in Ticker Method

```python
import pandas_ta_classic as ta

# Create empty DataFrame
df = pd.DataFrame()

# Fetch data (requires yfinance)
df = df.ta.ticker("AAPL", period="1y")

# Add indicators
df.ta.sma(length=20, append=True)
df.ta.ema(length=20, append=True)
```

### From CSV Files

```python
import pandas as pd
import pandas_ta_classic as ta

# Load CSV with OHLCV data
df = pd.read_csv('stock_data.csv', parse_dates=['date'], index_col='date')

# Ensure column names are lowercase (optional but recommended)
df.columns = [col.lower() for col in df.columns]

# Calculate indicators
df.ta.sma(length=20, append=True)
df.ta.atr(length=14, append=True)  # Average True Range
```

## 💡 Common Use Cases

### 1. Trend Following

Identify the trend direction:

```python
# Moving averages for trend
df.ta.sma(length=20, append=True)   # Short-term
df.ta.sma(length=50, append=True)   # Medium-term  
df.ta.sma(length=200, append=True)  # Long-term

# Trend indicator
df.ta.adx(length=14, append=True)   # Average Directional Index
```

### 2. Momentum Trading

Identify overbought/oversold conditions:

```python
# RSI for momentum
df.ta.rsi(length=14, append=True)

# Stochastic Oscillator
df.ta.stoch(append=True)

# Williams %R
df.ta.willr(length=14, append=True)
```

### 3. Volatility Analysis

Measure market volatility:

```python
# Bollinger Bands
df.ta.bbands(length=20, std=2, append=True)

# Average True Range
df.ta.atr(length=14, append=True)

# Keltner Channels
df.ta.kc(length=20, append=True)
```

### 4. Volume Analysis

Analyze trading volume:

```python
# On-Balance Volume
df.ta.obv(append=True)

# Volume Weighted Average Price
df.ta.vwap(append=True)

# Accumulation/Distribution
df.ta.ad(append=True)
```

### 5. Support & Resistance

Find key price levels:

```python
# Pivot Points
df.ta.pivot(append=True)

# Fibonacci Retracement (on a subset)
fib = ta.fib(df['high'], df['low'])
```

## 🎓 Understanding Indicator Output

### Single Column Output

Some indicators return a single Series:

```python
# Returns a Series
sma = df.ta.sma(length=20)
print(type(sma))  # pandas.Series
print(sma.name)   # 'SMA_20'
```

### Multiple Column Output

Others return a DataFrame with multiple columns:

```python
# Returns a DataFrame
bbands = df.ta.bbands(length=20)
print(type(bbands))     # pandas.DataFrame
print(bbands.columns)   # ['BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0']

# Access individual bands
lower_band = bbands['BBL_20_2.0']
middle_band = bbands['BBM_20_2.0']
upper_band = bbands['BBU_20_2.0']
```

### Custom Column Names

Use suffixes for clarity:

```python
# Add custom suffix
df.ta.sma(length=20, suffix="Daily", append=True)
# Creates column: SMA_20_Daily

# Calculate on different prices
df.ta.ema(close=df['high'], length=20, suffix="High", append=True)
# Creates column: EMA_20_High
```

## 🔍 Exploring Available Indicators

### List All Indicators

```python
import pandas_ta_classic as ta

# See all indicator categories
print(ta.Category)

# List indicators by category
print(ta.momentum)  # All momentum indicators
print(ta.overlap)   # All overlap indicators
print(ta.trend)     # All trend indicators
print(ta.volatility)  # All volatility indicators
print(ta.volume)    # All volume indicators
```

### Get Help on Indicators

```python
# Get indicator documentation
help(ta.sma)
help(ta.macd)
help(ta.rsi)

# Or use Python's built-in
print(ta.sma.__doc__)
```

## ⚡ Performance Tips

### 1. Use append=True Carefully

```python
# Good: Calculate once, append once
indicators = df.ta.sma(length=20)
df = pd.concat([df, indicators], axis=1)

# Better: Use append=True for convenience
df.ta.sma(length=20, append=True)

# Best: Use strategies for multiple indicators
df.ta.strategy("CommonStrategy")
```

### 2. Strategies Use Multiprocessing

```python
# Automatically parallelized
my_strategy = ta.Strategy(
    name="FastStrategy",
    ta=[
        {"kind": "sma", "length": 20},
        {"kind": "ema", "length": 20},
        {"kind": "rsi", "length": 14},
        {"kind": "macd"},
    ]
)
df.ta.strategy(my_strategy)  # Runs in parallel
```

### 3. Preprocess Data

```python
# Ensure data types are correct
df['close'] = pd.to_numeric(df['close'], errors='coerce')
df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

# Remove NaN values
df = df.dropna()
```

## 🐛 Troubleshooting

### Issue: "KeyError: 'close'"

**Solution:** Ensure your DataFrame has the required columns (open, high, low, close, volume):

```python
# Check column names
print(df.columns)

# Rename columns if needed
df.columns = ['open', 'high', 'low', 'close', 'volume']

# Or specify columns explicitly
df.ta.sma(close=df['price'], length=20)
```

### Issue: "All NaN values"

**Solution:** Check if you have enough data points:

```python
# Most indicators need minimum data
print(len(df))  # Should be > indicator length

# Example: SMA(20) needs at least 20 data points
if len(df) >= 20:
    df.ta.sma(length=20, append=True)
```

### Issue: "Import Error: No module named 'pandas_ta_classic'"

**Solution:** Verify installation:

```bash
pip list | grep pandas-ta-classic
# or
pip install --upgrade pandas-ta-classic
```

### Issue: TA-Lib candlestick patterns not available

**Solution:** Install TA-Lib separately:

```bash
# On Linux/Mac
pip install TA-Lib

# On Windows, download wheel from:
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib‑0.4.XX‑cpXX‑cpXX‑win_amd64.whl
```

## 📚 Next Steps

Now that you've got the basics, explore more:

1. **[Full Documentation](https://xgboosted.github.io/pandas-ta-classic/)** - Complete API reference
2. **[Tutorials](https://github.com/xgboosted/pandas-ta-classic/blob/main/TUTORIALS.md)** - Step-by-step guides for common tasks
3. **[Indicator Reference](https://xgboosted.github.io/pandas-ta-classic/indicators.html)** - All 203 indicators
4. **[Strategy Guide](https://xgboosted.github.io/pandas-ta-classic/strategies.html)** - Advanced strategy system
5. **[Examples](https://github.com/xgboosted/pandas-ta-classic/tree/main/examples)** - Jupyter notebooks with real examples

## 🎯 Quick Reference Card

| Task | Code |
|------|------|
| Install | `pip install pandas-ta-classic` |
| Import | `import pandas_ta_classic as ta` |
| Simple indicator | `df.ta.sma(length=20, append=True)` |
| Multiple indicators | `df.ta.strategy("CommonStrategy")` |
| Custom strategy | `ta.Strategy(name="My", ta=[...])` |
| List categories | `print(ta.Category)` |
| Get help | `help(ta.sma)` |
| Fetch data | `df.ta.ticker("AAPL")` |

## 💬 Need Help?

- **Issues**: [GitHub Issues](https://github.com/xgboosted/pandas-ta-classic/issues)
- **Discussions**: [GitHub Discussions](https://github.com/xgboosted/pandas-ta-classic/discussions)
- **Examples**: Check the [examples directory](https://github.com/xgboosted/pandas-ta-classic/tree/main/examples)

Happy Trading! 📈
