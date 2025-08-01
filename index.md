---
layout: default
title: Pandas TA Classic - Technical Analysis Library
---

# Pandas TA Classic

**Pandas Technical Analysis (Pandas TA Classic)** is an easy to use library that leverages the Pandas package with more than 130 Indicators and Utility functions. This is the **classic/community maintained version** of the popular pandas-ta library.

![Example Chart](images/TA_Chart.png)

## Features

* 130+ indicators and utility functions
* Tightly correlated with TA-Lib indicators
* Supports both standalone and DataFrame extension usage
* Multiprocessing support via Strategy method
* Custom strategies and indicator chaining
* Performance metrics (BETA)

## Quick Installation

```bash
# Stable version
pip install pandas-ta-classic

# Latest version from GitHub
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

- **Candles** (64): Pattern recognition indicators
- **Momentum** (41): RSI, MACD, Stochastic, etc.
- **Overlap** (33): Moving averages, Bollinger Bands, etc.
- **Trend** (18): ADX, Aroon, Parabolic SAR, etc.
- **Volume** (15): OBV, Money Flow, etc.
- **Volatility** (14): ATR, Bollinger Bands, etc.
- **Statistics** (11): Z-Score, Standard Deviation, etc.
- **Performance** (3): Returns, Drawdown analysis
- **Cycles** (1): Even Better Sinewave
- **Utility** (5): Helper functions

## Support

- [GitHub Issues](https://github.com/xgboosted/pandas-ta-classic/issues)
- [Examples and Notebooks](https://github.com/xgboosted/pandas-ta-classic/tree/main/examples)

## License

This project is licensed under the MIT License.