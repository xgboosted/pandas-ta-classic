---
layout: default
title: Pandas TA Classic - Technical Analysis Library
---

# Pandas TA Classic

**Pandas Technical Analysis (Pandas TA Classic)** is an easy to use library that leverages the Pandas package with more than 141 Indicators and Utility functions plus 62 TA-Lib candlestick patterns (203 total). This is the **community maintained version** of the popular pandas-ta library.

![Example Chart](images/TA_Chart.png)

## Features

* 141 indicators and utility functions across 9 categories
* 62 TA-Lib candlestick patterns for comprehensive pattern recognition
* Dynamic category discovery - automatically detects available indicators
* Tightly correlated with TA-Lib indicators
* Supports both standalone and DataFrame extension usage
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

# Or use strategies for bulk processing
df.ta.strategy("CommonStrategy")
```

## Documentation

For detailed documentation, examples, and the complete list of indicators, please visit our [GitHub repository](https://github.com/xgboosted/pandas-ta-classic).

## Categories of Indicators

- **Candles** (65): Pattern recognition indicators (3 native + 62 TA-Lib patterns)
- **Momentum** (41): RSI, MACD, Stochastic, etc.
- **Overlap** (34): Moving averages, Bollinger Bands, etc.
- **Trend** (18): ADX, Aroon, Parabolic SAR, etc.
- **Volume** (15): OBV, Money Flow, etc.
- **Volatility** (14): ATR, Bollinger Bands, etc.
- **Statistics** (10): Z-Score, Standard Deviation, etc.
- **Performance** (3): Returns, Drawdown analysis
- **Cycles** (1): Even Better Sinewave
- **Utility** (10): Helper functions

## Support

- [GitHub Issues](https://github.com/xgboosted/pandas-ta-classic/issues)
- [Examples and Notebooks](https://github.com/xgboosted/pandas-ta-classic/tree/main/examples)

## License

This project is licensed under the MIT License.