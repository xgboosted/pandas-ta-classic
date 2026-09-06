# Integrating pandas-ta-classic with manifoldbt

`manifoldbt` is a backtesting engine with a Rust core and a Python API. Strategies are written as expressions over whole columns (`when(fast > slow, 1.0, 0.0)`) rather than as a `next()` callback, and the engine evaluates them over a bar store it owns. Because it never hands you a per-bar Python hook, `pandas-ta-classic` integrates through a **precompute-then-register** pattern: indicators are computed over the full Series, then registered as an *exogenous series* the engine ASOF-joins onto the bars.

```bash
pip install manifoldbt
```

---

## 1. The Integration Pattern

1. Compute indicators over the full DataFrame with pandas-ta-classic.
2. Collect them into a DataFrame carrying a `timestamp` column, and drop the warmup rows.
3. Register that frame with `register_exo()`.
4. Reference the columns in expressions with `exo("<series>", "<column>")`, and list the series in `BacktestConfig(exo_data=[...])`.

```python
import manifoldbt as mbt
import pandas_ta_classic as ta
from manifoldbt.expr import col, exo, lit, when

# Step 1 - precompute over the full Series
signals = pd.DataFrame(
    {
        'timestamp': df['timestamp'],
        'sma_fast': ta.sma(df['close'], length=10),
        'sma_slow': ta.sma(df['close'], length=20),
    }
).dropna()

# Step 2 - the engine needs a bar store; a DataFrame can be imported directly
store = mbt.import_dataframe(df, symbol='SYNTH', symbol_id=1, interval='1d', exchange='dataframe')

# Step 3 - hand the indicators over as an exogenous series
mbt.register_exo('pandas_ta', signals, store=store)

# Step 4 - reference them in an expression
position = when(exo('pandas_ta', 'sma_fast') > exo('pandas_ta', 'sma_slow'), lit(1.0), lit(0.0))
strategy = mbt.Strategy.create('sma-crossover').signal('pos', position).size(col('pos'))
```

The bar DataFrame passed to `import_dataframe()` needs lowercase `timestamp,open,high,low,close,volume` columns, and the timestamps must be timezone-aware.

> **Why register instead of feeding a column?** The engine loads its bars from its own store, so an extra column on the input DataFrame is not visible to expressions. An exogenous series is the supported channel for values computed elsewhere. It is ASOF-joined and forward-filled onto bar timestamps, which is also what makes it safe to register a frame that is shorter than the bar history.

---

## 2. Common Integration Patterns

### Pattern A: Single-Output Indicators

Single-output indicators (SMA, RSI, EMA) return one Series. Give each one a column in the exo frame.

```python
signals = pd.DataFrame(
    {
        'timestamp': df['timestamp'],
        'rsi': ta.rsi(df['close'], length=14),
    }
).dropna()

mbt.register_exo('pandas_ta', signals, store=store)
position = when(exo('pandas_ta', 'rsi') < lit(30.0), lit(1.0), lit(0.0))
```

### Pattern B: Multi-Output Indicators (MACD, Bollinger Bands)

Multi-output indicators return a DataFrame whose column names carry the parameters (`ta.macd(...)` gives `MACD_12_26_9`, `MACDh_12_26_9`, `MACDs_12_26_9`). Pick the components you need and name them yourself: the exo column names are the ones you will type in the expression, so short stable names survive a parameter change.

```python
macd = ta.macd(df['close'], fast=12, slow=26, signal=9)

signals = pd.DataFrame(
    {
        'timestamp': df['timestamp'],
        'macd': macd['MACD_12_26_9'],
        'macd_signal': macd['MACDs_12_26_9'],
    }
).dropna()

position = when(exo('pandas_ta', 'macd') > exo('pandas_ta', 'macd_signal'), lit(1.0), lit(0.0))
```

### Pattern C: OHLCV-Dependent Indicators (ATR)

Indicators that need more than the close take the other Series as usual. Nothing changes on the engine side: the result is one more exo column.

```python
signals['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
```

Engine expressions compose on top of an exo column like any other series, so a volatility filter can be written against it directly:

```python
wide = exo('pandas_ta', 'atr') > exo('pandas_ta', 'atr').rolling_mean(50)
position = when(trend, when(wide, lit(0.5), lit(1.0)), lit(0.0))
```

---

## 3. Things That Bite

- **Declare the series in the config.** `BacktestConfig(exo_data=['pandas_ta'])` is what makes the columns resolvable; without it the expression refers to a column the engine never loaded.
- **Drop the warmup rows before registering.** A leading `NaN` block is forward-filled from nothing, so the first bars would compare against missing values. `dropna()` on the exo frame, and `warmup_bars` at least as long as the longest indicator window.
- **Name the columns in the exo frame, not in the expression.** `exo('pandas_ta', 'macd')` reads better than the parameter-encoded `MACD_12_26_9`, and it does not change when you retune the indicator.
- **A store keeps its metadata database open.** Removing a temporary store directory inside a `TemporaryDirectory` context manager fails on Windows while the process is alive; delete it with errors ignored, or keep the store on disk between runs.

---

## 4. Runnable Example

`examples/manifoldbt_strategy.py` runs the same SMA crossover as the backtesting.py and backtrader examples, on the same synthetic series, and prints total return, Sharpe, max drawdown and the round-trip count.

```bash
pip install manifoldbt
python examples/manifoldbt_strategy.py
```
