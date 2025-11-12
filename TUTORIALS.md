# 📚 Pandas TA Classic - Tutorials

Step-by-step tutorials for common workflows with **Pandas TA Classic**.

## 📋 Table of Contents

- [Tutorial 1: Creating a Moving Average Crossover Strategy](#tutorial-1-creating-a-moving-average-crossover-strategy)
- [Tutorial 2: Building a Custom Indicator Strategy](#tutorial-2-building-a-custom-indicator-strategy)
- [Tutorial 3: Backtesting with Performance Metrics](#tutorial-3-backtesting-with-performance-metrics)
- [Tutorial 4: Integrating with VectorBT](#tutorial-4-integrating-with-vectorbt)
- [Tutorial 5: Multi-Timeframe Analysis](#tutorial-5-multi-timeframe-analysis)
- [Tutorial 6: Creating Custom Indicators](#tutorial-6-creating-custom-indicators)
- [Tutorial 7: Candlestick Pattern Recognition](#tutorial-7-candlestick-pattern-recognition)

---

## Tutorial 1: Creating a Moving Average Crossover Strategy

Learn to create a classic moving average crossover trading strategy.

### Goal
Identify buy and sell signals when a fast moving average crosses above/below a slow moving average.

### Step 1: Set Up Your Environment

```python
import pandas as pd
import pandas_ta_classic as ta
import yfinance as yf
import matplotlib.pyplot as plt

# Download data
df = yf.download("AAPL", start="2023-01-01", end="2024-01-01")
print(f"Downloaded {len(df)} rows of data")
```

### Step 2: Calculate Moving Averages

```python
# Calculate fast and slow moving averages
df.ta.sma(length=20, append=True, col_names=("SMA_fast",))
df.ta.sma(length=50, append=True, col_names=("SMA_slow",))

# Preview the data
print(df[['Close', 'SMA_fast', 'SMA_slow']].tail())
```

### Step 3: Generate Trading Signals

```python
# Create signal column
df['signal'] = 0

# Buy signal: fast MA crosses above slow MA
df.loc[df['SMA_fast'] > df['SMA_slow'], 'signal'] = 1

# Sell signal: fast MA crosses below slow MA
df.loc[df['SMA_fast'] < df['SMA_slow'], 'signal'] = -1

# Identify crossover points
df['position'] = df['signal'].diff()

print("Buy signals:", len(df[df['position'] == 2]))
print("Sell signals:", len(df[df['position'] == -2]))
```

### Step 4: Visualize the Strategy

```python
plt.figure(figsize=(14, 7))

# Plot price and moving averages
plt.plot(df.index, df['Close'], label='Close Price', alpha=0.5)
plt.plot(df.index, df['SMA_fast'], label='SMA 20 (Fast)', alpha=0.7)
plt.plot(df.index, df['SMA_slow'], label='SMA 50 (Slow)', alpha=0.7)

# Plot buy signals
buy_signals = df[df['position'] == 2]
plt.scatter(buy_signals.index, buy_signals['Close'], 
            color='green', marker='^', s=100, label='Buy Signal')

# Plot sell signals
sell_signals = df[df['position'] == -2]
plt.scatter(sell_signals.index, sell_signals['Close'], 
            color='red', marker='v', s=100, label='Sell Signal')

plt.title('Moving Average Crossover Strategy - AAPL')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Key Takeaways
- Moving average crossovers are a fundamental trading strategy
- Use `diff()` to detect actual crossover events
- Always visualize your strategy before trading

---

## Tutorial 2: Building a Custom Indicator Strategy

Create a comprehensive strategy using multiple indicators.

### Goal
Build a strategy combining trend, momentum, and volatility indicators.

### Step 1: Define Your Strategy

```python
import pandas as pd
import pandas_ta_classic as ta
import yfinance as yf

# Download data
df = yf.download("SPY", start="2023-01-01", end="2024-01-01")

# Create custom strategy
my_strategy = ta.Strategy(
    name="TrendMomentumVolatility",
    description="Combines trend, momentum, and volatility indicators",
    ta=[
        # Trend Indicators
        {"kind": "sma", "length": 20},
        {"kind": "sma", "length": 50},
        {"kind": "ema", "length": 12},
        {"kind": "ema", "length": 26},
        
        # Momentum Indicators
        {"kind": "rsi", "length": 14},
        {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
        {"kind": "stoch", "k": 14, "d": 3},
        
        # Volatility Indicators
        {"kind": "bbands", "length": 20, "std": 2},
        {"kind": "atr", "length": 14},
        {"kind": "kc", "length": 20},
    ]
)

print(f"Strategy: {my_strategy.name}")
print(f"Number of indicators: {len(my_strategy.ta)}")
```

### Step 2: Run the Strategy

```python
# Execute the strategy (uses multiprocessing)
df.ta.strategy(my_strategy)

# Check new columns
new_columns = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
print(f"\nAdded {len(new_columns)} indicator columns:")
print(new_columns)
```

### Step 3: Create Trading Rules

```python
# Define bullish conditions
bullish_conditions = (
    (df['Close'] > df['SMA_20']) &           # Price above 20 SMA
    (df['SMA_20'] > df['SMA_50']) &          # Short MA above long MA
    (df['RSI_14'] > 30) & (df['RSI_14'] < 70) &  # RSI in neutral zone
    (df['Close'] > df['BBL_20_2.0'])         # Price above lower Bollinger Band
)

# Define bearish conditions
bearish_conditions = (
    (df['Close'] < df['SMA_20']) &           # Price below 20 SMA
    (df['SMA_20'] < df['SMA_50']) &          # Short MA below long MA
    (df['RSI_14'] > 30) & (df['RSI_14'] < 70) &  # RSI in neutral zone
    (df['Close'] < df['BBU_20_2.0'])         # Price below upper Bollinger Band
)

# Create signals
df['strategy_signal'] = 0
df.loc[bullish_conditions, 'strategy_signal'] = 1
df.loc[bearish_conditions, 'strategy_signal'] = -1

print(f"Bullish signals: {(df['strategy_signal'] == 1).sum()}")
print(f"Bearish signals: {(df['strategy_signal'] == -1).sum()}")
```

### Step 4: Analyze Signal Quality

```python
# Calculate signal strength
df['signal_strength'] = 0

# Add points for each bullish condition
df.loc[df['Close'] > df['SMA_20'], 'signal_strength'] += 1
df.loc[df['SMA_20'] > df['SMA_50'], 'signal_strength'] += 1
df.loc[df['RSI_14'] > 50, 'signal_strength'] += 1
df.loc[df['MACDh_12_26_9'] > 0, 'signal_strength'] += 1

# Filter for strong signals only
strong_bullish = df[(df['strategy_signal'] == 1) & (df['signal_strength'] >= 3)]
print(f"Strong bullish signals: {len(strong_bullish)}")
```

### Key Takeaways
- Combine multiple indicators for confirmation
- Use multiprocessing with Strategy class
- Filter signals by strength for better quality

---

## Tutorial 3: Backtesting with Performance Metrics

Calculate returns and performance metrics for your strategy.

### Goal
Evaluate strategy performance using pandas-ta-classic's performance utilities.

### Step 1: Calculate Strategy Returns

```python
import pandas as pd
import pandas_ta_classic as ta
import yfinance as yf
import numpy as np

# Get data and add indicators
df = yf.download("AAPL", start="2022-01-01", end="2024-01-01")
df.ta.sma(length=20, append=True)
df.ta.sma(length=50, append=True)

# Generate signals
df['position'] = 0
df.loc[df['SMA_20'] > df['SMA_50'], 'position'] = 1  # Long position

# Calculate returns
df['returns'] = df['Close'].pct_change()
df['strategy_returns'] = df['position'].shift(1) * df['returns']

# Cumulative returns
df['cumulative_returns'] = (1 + df['returns']).cumprod()
df['cumulative_strategy_returns'] = (1 + df['strategy_returns']).cumprod()

print(f"Buy and Hold Return: {(df['cumulative_returns'].iloc[-1] - 1) * 100:.2f}%")
print(f"Strategy Return: {(df['cumulative_strategy_returns'].iloc[-1] - 1) * 100:.2f}%")
```

### Step 2: Calculate Performance Metrics

```python
# Use pandas-ta-classic's performance utilities
from pandas_ta_classic import (
    percent_return, 
    log_return, 
    cagr,
    calmar_ratio,
    sharpe_ratio,
    sortino_ratio
)

# Calculate metrics on strategy equity curve
strategy_equity = df['Close'] * df['cumulative_strategy_returns'] / df['cumulative_returns']

# CAGR (Compound Annual Growth Rate)
strategy_cagr = ta.cagr(strategy_equity)
print(f"\nCAGR: {strategy_cagr * 100:.2f}%")

# Sharpe Ratio
strategy_sharpe = ta.sharpe_ratio(strategy_equity)
print(f"Sharpe Ratio: {strategy_sharpe:.2f}")

# Calmar Ratio
strategy_calmar = ta.calmar_ratio(strategy_equity)
print(f"Calmar Ratio: {strategy_calmar:.2f}")
```

### Step 3: Calculate Drawdown

```python
# Maximum drawdown
running_max = df['cumulative_strategy_returns'].expanding().max()
drawdown = (df['cumulative_strategy_returns'] - running_max) / running_max
max_drawdown = drawdown.min()

print(f"\nMax Drawdown: {max_drawdown * 100:.2f}%")

# Visualize drawdown
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Plot cumulative returns
ax1.plot(df.index, df['cumulative_returns'], label='Buy & Hold', linewidth=2)
ax1.plot(df.index, df['cumulative_strategy_returns'], label='Strategy', linewidth=2)
ax1.set_title('Cumulative Returns Comparison')
ax1.set_ylabel('Cumulative Return')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot drawdown
ax2.fill_between(df.index, 0, drawdown * 100, color='red', alpha=0.3)
ax2.set_title('Strategy Drawdown')
ax2.set_xlabel('Date')
ax2.set_ylabel('Drawdown (%)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Step 4: Win Rate Analysis

```python
# Calculate win rate
winning_trades = df[df['strategy_returns'] > 0]
losing_trades = df[df['strategy_returns'] < 0]

win_rate = len(winning_trades) / (len(winning_trades) + len(losing_trades)) * 100
avg_win = winning_trades['strategy_returns'].mean() * 100
avg_loss = losing_trades['strategy_returns'].mean() * 100

print(f"\n{'='*50}")
print(f"Win Rate: {win_rate:.2f}%")
print(f"Average Win: {avg_win:.2f}%")
print(f"Average Loss: {avg_loss:.2f}%")
print(f"Profit Factor: {abs(avg_win / avg_loss):.2f}")
print(f"{'='*50}")
```

### Key Takeaways
- Always calculate multiple performance metrics
- Consider risk-adjusted returns (Sharpe, Sortino)
- Analyze drawdowns to understand risk
- Track win rate and average win/loss

---

## Tutorial 4: Integrating with VectorBT

Use pandas-ta-classic indicators with the VectorBT backtesting framework.

### Goal
Backtest strategies efficiently using VectorBT's vectorized approach.

### Step 1: Install and Import VectorBT

```bash
pip install vectorbt
```

```python
import pandas as pd
import pandas_ta_classic as ta
import vectorbt as vbt
import yfinance as yf

# Download data
df = yf.download("AAPL", start="2022-01-01", end="2024-01-01")
```

### Step 2: Calculate Indicators

```python
# Add your indicators using pandas-ta-classic
df.ta.sma(length=20, append=True)
df.ta.sma(length=50, append=True)
df.ta.rsi(length=14, append=True)
df.ta.atr(length=14, append=True)

print(df[['Close', 'SMA_20', 'SMA_50', 'RSI_14']].tail())
```

### Step 3: Create Entry and Exit Signals

```python
# Generate entry signals (MA crossover + RSI confirmation)
entries = (
    (df['SMA_20'] > df['SMA_50']) &  # Fast MA above slow MA
    (df['RSI_14'] > 30) &             # RSI above oversold
    (df['RSI_14'] < 70)               # RSI below overbought
)

# Generate exit signals
exits = (
    (df['SMA_20'] < df['SMA_50']) |   # Fast MA below slow MA
    (df['RSI_14'] > 80)                # RSI extremely overbought
)
```

### Step 4: Run VectorBT Backtest

```python
# Create portfolio
portfolio = vbt.Portfolio.from_signals(
    df['Close'],
    entries,
    exits,
    init_cash=10000,
    fees=0.001,  # 0.1% trading fees
    freq='1D'
)

# Display results
print(portfolio.stats())
print(f"\nTotal Return: {portfolio.total_return() * 100:.2f}%")
print(f"Sharpe Ratio: {portfolio.sharpe_ratio():.2f}")
print(f"Max Drawdown: {portfolio.max_drawdown() * 100:.2f}%")
```

### Step 5: Visualize Results

```python
# Plot portfolio value
portfolio.plot().show()

# Plot trades on price chart
fig = df['Close'].vbt.plot()
portfolio.positions.plot(trace_kwargs=dict(name='Position')).show()
```

### Key Takeaways
- VectorBT handles the heavy lifting of backtesting
- Use pandas-ta-classic for indicator calculation
- VectorBT provides detailed performance analytics
- Easy to test multiple parameter combinations

---

## Tutorial 5: Multi-Timeframe Analysis

Analyze different timeframes to confirm trends.

### Goal
Combine indicators from multiple timeframes for better signal quality.

### Step 1: Download Multiple Timeframes

```python
import pandas as pd
import pandas_ta_classic as ta
import yfinance as yf

# Download different timeframes
ticker = "SPY"
df_daily = yf.download(ticker, period="1y", interval="1d")
df_hourly = yf.download(ticker, period="60d", interval="1h")
df_5min = yf.download(ticker, period="5d", interval="5m")

print(f"Daily: {len(df_daily)} bars")
print(f"Hourly: {len(df_hourly)} bars")
print(f"5-min: {len(df_5min)} bars")
```

### Step 2: Calculate Indicators for Each Timeframe

```python
# Daily timeframe - trend
df_daily.ta.sma(length=50, append=True)
df_daily.ta.sma(length=200, append=True)
df_daily['daily_trend'] = 'neutral'
df_daily.loc[df_daily['SMA_50'] > df_daily['SMA_200'], 'daily_trend'] = 'bullish'
df_daily.loc[df_daily['SMA_50'] < df_daily['SMA_200'], 'daily_trend'] = 'bearish'

# Hourly timeframe - momentum
df_hourly.ta.rsi(length=14, append=True)
df_hourly.ta.macd(append=True)

# 5-minute timeframe - entry timing
df_5min.ta.sma(length=20, append=True)
df_5min.ta.bbands(length=20, append=True)
```

### Step 3: Align Timeframes

```python
# Resample higher timeframe data to lower timeframe
# Forward fill to propagate daily trend to hourly
df_hourly['date'] = df_hourly.index.date
df_daily['date'] = df_daily.index.date

# Merge daily trend into hourly data
df_hourly = df_hourly.merge(
    df_daily[['date', 'daily_trend']], 
    on='date', 
    how='left'
)

print("\nHourly data with daily trend:")
print(df_hourly[['Close', 'RSI_14', 'daily_trend']].tail())
```

### Step 4: Create Multi-Timeframe Signals

```python
# Only take trades aligned with higher timeframe
# Bullish hourly signal when daily trend is bullish
hourly_bullish = (
    (df_hourly['daily_trend'] == 'bullish') &  # Daily uptrend
    (df_hourly['RSI_14'] < 40) &                # Hourly oversold
    (df_hourly['MACDh_12_26_9'] > 0)           # MACD histogram positive
)

# Bearish hourly signal when daily trend is bearish  
hourly_bearish = (
    (df_hourly['daily_trend'] == 'bearish') &  # Daily downtrend
    (df_hourly['RSI_14'] > 60) &                # Hourly overbought
    (df_hourly['MACDh_12_26_9'] < 0)           # MACD histogram negative
)

df_hourly['mtf_signal'] = 0
df_hourly.loc[hourly_bullish, 'mtf_signal'] = 1
df_hourly.loc[hourly_bearish, 'mtf_signal'] = -1

print(f"\nBullish signals: {(df_hourly['mtf_signal'] == 1).sum()}")
print(f"Bearish signals: {(df_hourly['mtf_signal'] == -1).sum()}")
```

### Key Takeaways
- Higher timeframes provide trend direction
- Lower timeframes provide entry timing
- Only trade in the direction of higher timeframe trend
- Reduces false signals and improves win rate

---

## Tutorial 6: Creating Custom Indicators

Build your own custom indicators using pandas-ta-classic as a foundation.

### Goal
Create a custom indicator that combines existing indicators.

### Step 1: Define Your Custom Indicator

```python
import pandas as pd
import pandas_ta_classic as ta
import numpy as np

def trend_strength(high, low, close, length=14):
    """
    Custom Trend Strength Indicator
    Combines ADX, RSI, and price position in Bollinger Bands
    Returns a score from -100 to +100
    """
    # Calculate components
    adx_df = ta.adx(high, low, close, length=length)
    rsi = ta.rsi(close, length=length)
    bbands = ta.bbands(close, length=length)
    
    # ADX component (0 to 25 scale)
    adx_score = (adx_df[f'ADX_{length}'] / 100) * 25
    
    # RSI component (-25 to +25 scale)
    rsi_score = ((rsi - 50) / 50) * 25
    
    # Bollinger Band position component (-50 to +50 scale)
    bb_position = (close - bbands[f'BBM_{length}_2.0']) / (bbands[f'BBU_{length}_2.0'] - bbands[f'BBL_{length}_2.0'])
    bb_score = bb_position * 50
    
    # Combine scores
    trend_score = adx_score + rsi_score + bb_score
    
    # Create result DataFrame
    result = pd.DataFrame({
        f'TREND_STRENGTH_{length}': trend_score,
        f'TS_ADX_{length}': adx_score,
        f'TS_RSI_{length}': rsi_score,
        f'TS_BB_{length}': bb_score
    }, index=close.index)
    
    return result
```

### Step 2: Test Your Custom Indicator

```python
import yfinance as yf

# Download data
df = yf.download("AAPL", period="6mo")

# Apply custom indicator
trend_score = trend_strength(df['High'], df['Low'], df['Close'], length=14)

# Add to DataFrame
df = pd.concat([df, trend_score], axis=1)

print(df[['Close', 'TREND_STRENGTH_14', 'TS_ADX_14', 'TS_RSI_14', 'TS_BB_14']].tail())
```

### Step 3: Visualize Your Indicator

```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Plot price
ax1.plot(df.index, df['Close'], label='Close Price')
ax1.set_ylabel('Price ($)')
ax1.set_title('AAPL Price with Custom Trend Strength Indicator')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot custom indicator
ax2.plot(df.index, df['TREND_STRENGTH_14'], label='Trend Strength', linewidth=2)
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax2.axhline(y=50, color='green', linestyle=':', linewidth=1, alpha=0.5)
ax2.axhline(y=-50, color='red', linestyle=':', linewidth=1, alpha=0.5)
ax2.fill_between(df.index, 0, df['TREND_STRENGTH_14'], 
                  where=(df['TREND_STRENGTH_14'] > 0), color='green', alpha=0.3)
ax2.fill_between(df.index, 0, df['TREND_STRENGTH_14'], 
                  where=(df['TREND_STRENGTH_14'] < 0), color='red', alpha=0.3)
ax2.set_ylabel('Trend Strength')
ax2.set_xlabel('Date')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Step 4: Use in Trading Strategy

```python
# Generate signals based on custom indicator
df['signal'] = 0
df.loc[df['TREND_STRENGTH_14'] > 50, 'signal'] = 1   # Strong bullish
df.loc[df['TREND_STRENGTH_14'] < -50, 'signal'] = -1  # Strong bearish

# Count signals
print(f"Bullish signals: {(df['signal'] == 1).sum()}")
print(f"Bearish signals: {(df['signal'] == -1).sum()}")
```

### Key Takeaways
- Build on existing pandas-ta-classic indicators
- Normalize different indicators to same scale
- Test thoroughly before live trading
- Visualize to understand indicator behavior

---

## Tutorial 7: Candlestick Pattern Recognition

Identify Japanese candlestick patterns for trading signals.

### Goal
Use candlestick patterns to enhance entry and exit timing.

### Step 1: Setup (requires TA-Lib)

```python
import pandas as pd
import pandas_ta_classic as ta
import yfinance as yf

# Download data
df = yf.download("SPY", period="3mo")

# Calculate baseline indicators
df.ta.sma(length=50, append=True)
df.ta.rsi(length=14, append=True)
```

### Step 2: Detect Candlestick Patterns

```python
# Pandas TA Classic native patterns (no TA-Lib required)
df.ta.cdl_doji(append=True)        # Doji pattern
df.ta.cdl_inside(append=True)      # Inside bar
df.ta.ha(append=True)              # Heikin Ashi

# If TA-Lib is installed, you can use all patterns
# Example: Hammer, Shooting Star, Engulfing, etc.
if ta.Imports['talib']:
    df.ta.cdl_pattern(name="hammer", append=True)
    df.ta.cdl_pattern(name="engulfing", append=True)
    df.ta.cdl_pattern(name="morningstar", append=True)
    df.ta.cdl_pattern(name="eveningstar", append=True)

print("Patterns detected:")
pattern_cols = [col for col in df.columns if col.startswith('CDL_')]
print(pattern_cols)
```

### Step 3: Combine Patterns with Trend

```python
# Only trade patterns in direction of trend
uptrend = df['Close'] > df['SMA_50']
downtrend = df['Close'] < df['SMA_50']

# Bullish setups: bullish pattern + uptrend + oversold RSI
if 'CDL_HAMMER' in df.columns:
    bullish_setup = (
        (df['CDL_HAMMER'] > 0) &  # Hammer pattern detected
        uptrend &                  # In uptrend
        (df['RSI_14'] < 50)       # RSI in lower half
    )
    
    df['bullish_pattern_signal'] = bullish_setup.astype(int)
    print(f"Bullish pattern signals: {bullish_setup.sum()}")

# Bearish setups: bearish pattern + downtrend + overbought RSI
if 'CDL_SHOOTINGSTAR' in df.columns:
    bearish_setup = (
        (df['CDL_SHOOTINGSTAR'] > 0) &  # Shooting star detected
        downtrend &                      # In downtrend
        (df['RSI_14'] > 50)             # RSI in upper half
    )
    
    df['bearish_pattern_signal'] = bearish_setup.astype(int)
    print(f"Bearish pattern signals: {bearish_setup.sum()}")
```

### Step 4: Pattern Statistics

```python
# Analyze pattern effectiveness
if 'CDL_DOJI' in df.columns:
    doji_days = df[df['CDL_DOJI'] != 0].copy()
    
    # What happens 5 days after a doji?
    doji_days['future_return'] = df['Close'].shift(-5) / df['Close'] - 1
    
    print("\nDoji Pattern Analysis:")
    print(f"Total Doji patterns: {len(doji_days)}")
    print(f"Average return after 5 days: {doji_days['future_return'].mean() * 100:.2f}%")
    print(f"Win rate: {(doji_days['future_return'] > 0).sum() / len(doji_days) * 100:.2f}%")
```

### Step 5: Visualize Patterns

```python
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Create candlestick chart with pattern markers
fig, ax = plt.subplots(figsize=(14, 7))

# Plot closing price
ax.plot(df.index, df['Close'], label='Close', linewidth=1, alpha=0.7)
ax.plot(df.index, df['SMA_50'], label='SMA 50', linewidth=2, alpha=0.7)

# Mark pattern occurrences
if 'CDL_DOJI' in df.columns:
    doji_mask = df['CDL_DOJI'] != 0
    ax.scatter(df[doji_mask].index, df[doji_mask]['Close'], 
               color='blue', marker='o', s=100, label='Doji', zorder=5)

if 'CDL_HAMMER' in df.columns:
    hammer_mask = df['CDL_HAMMER'] > 0
    ax.scatter(df[hammer_mask].index, df[hammer_mask]['Close'], 
               color='green', marker='^', s=150, label='Hammer', zorder=5)

ax.set_title('Candlestick Patterns on SPY')
ax.set_xlabel('Date')
ax.set_ylabel('Price ($)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Key Takeaways
- Candlestick patterns work best with trend confirmation
- Combine patterns with momentum indicators
- Backtest pattern effectiveness on your specific instrument
- Some patterns require TA-Lib, others are built-in

---

## 🎯 Next Steps

You've completed the tutorials! Here's what to explore next:

1. **Experiment with Different Timeframes** - Test strategies on various intervals
2. **Optimize Parameters** - Use grid search to find best indicator settings
3. **Paper Trade** - Test strategies in real-time without risk
4. **Read the Docs** - Deep dive into the [full documentation](https://xgboosted.github.io/pandas-ta-classic/)
5. **Join the Community** - Share your strategies in [Discussions](https://github.com/xgboosted/pandas-ta-classic/discussions)

## 📚 Additional Resources

- [QUICKSTART.md](QUICKSTART.md) - Quick reference guide
- [Examples Directory](examples/) - Jupyter notebooks
- [Indicator Reference](https://xgboosted.github.io/pandas-ta-classic/indicators.html) - All indicators
- [Strategy Guide](https://xgboosted.github.io/pandas-ta-classic/strategies.html) - Advanced strategies

Happy Trading! 📈
