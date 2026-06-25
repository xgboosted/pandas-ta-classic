# Integrating pandas-ta-classic with vectorbt

`vectorbt` is a vectorized backtesting library built on NumPy and pandas. Unlike event-driven frameworks, it operates on entire time-series at once, making parameter sweeps fast. Since `pandas-ta-classic` returns pandas Series and DataFrames natively, the two integrate without a bridge function.

The key connector is `df.ta.tsignals()`, which converts a boolean trend Series into vectorbt-compatible entry and exit boolean arrays.

---

## 1. The Integration Pattern

```python
import pandas as pd
import pandas_ta_classic as ta
import vectorbt as vbt
```

**Step 1 — Define a trend as a boolean Series:**

```python
def trend(df: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.Series:
    return ta.ma("sma", df["Close"], length=fast) > ta.ma("sma", df["Close"], length=slow)
```

A trend is `True` when the condition holds, `False` otherwise. Any boolean expression over indicator output works here.

**Step 2 — Convert the trend to entry/exit signals with `tsignals`:**

```python
signals = df.ta.tsignals(trend(df), asbool=True, trade_offset=1)
# signals columns: TS_Trends, TS_Trades, TS_Entries, TS_Exits
```

- `asbool=True` — returns boolean arrays, which is what `vbt.Portfolio.from_signals()` expects.
- `trade_offset=1` — shifts entries/exits by one bar to avoid look-ahead bias in backtesting. Use `0` for live signals.

**Step 3 — Run the backtest:**

```python
vbt.settings.portfolio["freq"] = "1D"
vbt.settings.portfolio["fees"] = 0.0025
vbt.settings.portfolio["slippage"] = 0.0025

pf = vbt.Portfolio.from_signals(
    df["Close"],
    entries=signals.TS_Entries,
    exits=signals.TS_Exits,
)

print(pf.stats())
```

---

## 2. Comparing Against Buy-and-Hold

```python
pf_bnh = vbt.Portfolio.from_holding(df["Close"])

print("Strategy:")
print(pf.stats()[["Total Return [%]", "Sharpe Ratio", "Max Drawdown [%]"]])

print("\nBuy and Hold:")
print(pf_bnh.stats()[["Total Return [%]", "Sharpe Ratio", "Max Drawdown [%]"]])
```

---

## 3. Plotting

```python
pf.trades.plot(title="Trades").show()
pf.value().vbt.plot(title="Equity Curve").show()
pf.drawdown().vbt.plot(title="Drawdown").show()
```

---

## 4. Multi-Output Indicators

Indicators that return a DataFrame (MACD, Bollinger Bands) work directly — index them by column name before passing to the trend function:

```python
def macd_trend(df: pd.DataFrame) -> pd.Series:
    macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    return macd["MACDh_12_26_9"] > 0  # positive histogram = bullish
```

---

> **Full example:** See [`examples/VectorBT_Backtest_with_Pandas_TA.ipynb`](https://github.com/xgboosted/pandas-ta-classic/blob/main/examples/VectorBT_Backtest_with_Pandas_TA.ipynb) for a complete workflow including multi-ticker data acquisition, benchmark comparison, and full equity curve plots.
