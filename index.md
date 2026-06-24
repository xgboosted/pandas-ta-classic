---
layout: default
title: Pandas TA Classic - Technical Analysis Library
---

# Pandas TA Classic

**Pandas Technical Analysis (Pandas TA Classic)** is an easy to use library that leverages the Pandas package with **192 indicators and utility functions** plus **62 native candlestick patterns** (**252 total unique** — no TA-Lib required). This is the **community maintained version** of the popular pandas-ta library.

![Example Chart](https://raw.githubusercontent.com/xgboosted/pandas-ta-classic/main/docs/images/TA_Chart.png)

## Features

* **192 indicators and utility functions** across 9 categories
* **62 native candlestick patterns** — all natively implemented, no TA-Lib required
* **252 total unique indicators and patterns**
* Dynamic category discovery - automatically detects available indicators
* **Optional TA-Lib Acceleration**: 34 core indicators auto-use TA-Lib when installed; pass `talib=False` to force native
* **Optional Oracle Libraries**: TA-Lib (acceleration backend + oracle) and tulipy (oracle only) are optional and skip gracefully when not installed
* Supports both standalone and DataFrame extension usage
* **Fluent API chaining**: ``df.ta.chain().sma(20).ta.rsi(14).ta.macd()``
* Multiprocessing support via Strategy method
* Custom strategies and indicator chaining
* Performance metrics (BETA)

## Quick Installation

Supports both modern `uv` and traditional `pip`:

Using `uv` (recommended):
```bash
uv pip install pandas-ta-classic
```

Using `pip`:
```bash
pip install pandas-ta-classic
```

Latest version:
```bash
# Using uv
uv pip install git+https://github.com/xgboosted/pandas-ta-classic

# Using pip
pip install -U git+https://github.com/xgboosted/pandas-ta-classic
```

## Quick Start

```python
import pandas as pd
import pandas_ta_classic as ta

# Load your data
df = pd.read_csv("path/to/symbol.csv")

# Calculate indicators
df.ta.sma(length=20, append=True)  # Simple Moving Average
df.ta.rsi(length=14, append=True)  # RSI
df.ta.macd(append=True)            # MACD

# Fluent API chaining
df.ta.chain().sma(20).ta.rsi(14).ta.macd().ta.bbands(20)

# Or use strategies for bulk processing
df.ta.strategy("CommonStrategy")
```

## Documentation

For detailed documentation, examples, and the complete list of indicators, please visit our [GitHub repository](https://github.com/xgboosted/pandas-ta-classic).

## Categories of Indicators

- **Candles** (67): Candle-focused indicators, including 62 native CDL patterns (no TA-Lib required)
- **Momentum** (53): RSI, MACD, Stochastic, etc.
- **Overlap** (46): Moving averages, Bollinger Bands, etc.
- **Trend** (26): ADX, Aroon, Parabolic SAR, etc.
- **Volume** (20): OBV, Money Flow, etc.
- **Volatility** (18): ATR, Bollinger Bands, Chandelier Exit, etc.
- **Statistics** (14): Z-Score, Standard Deviation, etc.
- **Performance** (3): Returns, Drawdown analysis
- **Cycles** (8): Hilbert Transform and cycle detection
- **Math** (28): Element-wise math operators and transforms

## Support

- [GitHub Issues](https://github.com/xgboosted/pandas-ta-classic/issues)
- [Examples and Notebooks](https://github.com/xgboosted/pandas-ta-classic/tree/main/examples)

## License

This project is licensed under the MIT License.