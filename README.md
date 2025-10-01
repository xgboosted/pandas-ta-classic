<p align="center">
  <a href="https://github.com/xgboosted/pandas-ta-classic">
    <img src="https://raw.githubusercontent.com/xgboosted/pandas-ta-classic/main/images/logo.png" width="150" height="150" alt="Pandas TA Classic">
  </a>
</p>

# Pandas TA Classic - Technical Analysis Library

[![license](https://img.shields.io/github/license/xgboosted/pandas-ta-classic)](#license)
[![Build Status](https://github.com/xgboosted/pandas-ta-classic/workflows/CI/badge.svg)](https://github.com/xgboosted/pandas-ta-classic/actions)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://xgboosted.github.io/pandas-ta-classic/)
[![Python Version](https://img.shields.io/pypi/pyversions/pandas-ta-classic?style=flat)](https://pypi.org/project/pandas-ta-classic/)
[![PyPi Version](https://img.shields.io/pypi/v/pandas-ta-classic?style=flat)](https://pypi.org/project/pandas-ta-classic/)
[![Package Status](https://img.shields.io/pypi/status/pandas-ta-classic?style=flat)](https://pypi.org/project/pandas-ta-classic/)
[![Downloads](https://img.shields.io/pypi/dm/pandas-ta-classic?style=flat)](https://pypistats.org/packages/pandas-ta-classic)
[![Stars](https://img.shields.io/github/stars/xgboosted/pandas-ta-classic?style=flat)](#stars)
[![Forks](https://img.shields.io/github/forks/xgboosted/pandas-ta-classic?style=flat)](#forks)
[![Used By](https://img.shields.io/badge/used_by-170-orange.svg?style=flat)](#usedby)
[![Contributors](https://img.shields.io/github/contributors/xgboosted/pandas-ta-classic?style=flat)](#contributors)

![Example Chart](https://raw.githubusercontent.com/xgboosted/pandas-ta-classic/main/images/TA_Chart.png)

**Pandas TA Classic** is an easy-to-use library that leverages the Pandas package with **141 indicators and utility functions** and **62 TA Lib candlestick patterns** (**203 total**). Many commonly used indicators are included, such as: _Simple Moving Average_ (**sma**), _Moving Average Convergence Divergence_ (**macd**), _Hull Exponential Moving Average_ (**hma**), _Bollinger Bands_ (**bbands**), _On-Balance Volume_ (**obv**), _Aroon & Aroon Oscillator_ (**aroon**), _Squeeze_ (**squeeze**) and **many more**.

This is the **classic/community maintained version** of the popular pandas-ta library.

### 🎯 Key Features

- **203 Technical Indicators**: 141 indicators + 62 TA-Lib candlestick patterns
- **Native Candlestick Patterns**: 5 patterns (cdl_doji, cdl_inside, cdl_z, cdl_pattern, ha) work without TA-Lib
- **Automatic Versioning**: Version management via git tags using setuptools-scm
- **Modern Package Management**: Full support for both `uv` and `pip`
- **Production Ready**: Stable status with comprehensive test coverage
- **Active Development**: Regular updates with community contributions

## 🚀 Quick Start

### Installation

The library supports both modern **uv** and traditional **pip** package managers.

**Stable Release**

Using `uv` (recommended - faster):
```bash
uv pip install pandas-ta-classic
```

Using `pip`:
```bash
pip install pandas-ta-classic
```

**Latest Version**

Using `uv`:
```bash
uv pip install git+https://github.com/xgboosted/pandas-ta-classic
```

Using `pip`:
```bash
pip install -U git+https://github.com/xgboosted/pandas-ta-classic
```

**Development Installation**

Using `uv`:
```bash
# Clone the repository
git clone https://github.com/xgboosted/pandas-ta-classic.git
cd pandas-ta-classic

# Install with all dependencies
uv pip install -e ".[all]"

# Or install specific dependency groups:
uv pip install -e ".[dev]"      # Development tools
uv pip install -e ".[optional]" # Optional features like TA-Lib
```

Using `pip`:
```bash
# Clone the repository
git clone https://github.com/xgboosted/pandas-ta-classic.git
cd pandas-ta-classic

# Install with all dependencies
pip install -e ".[all]"

# Or install specific dependency groups:
pip install -e ".[dev]"      # Development tools
pip install -e ".[optional]" # Optional features like TA-Lib
```

### Basic Usage

```python
import pandas as pd
import pandas_ta_classic as ta

# Load your data
df = pd.read_csv("path/to/symbol.csv")
# OR if you have yfinance installed
df = df.ta.ticker("aapl")

# Calculate indicators
df.ta.sma(length=20, append=True)        # Simple Moving Average
df.ta.rsi(append=True)                   # Relative Strength Index  
df.ta.macd(append=True)                  # MACD
df.ta.bbands(append=True)                # Bollinger Bands

# Or run a strategy with multiple indicators
df.ta.strategy("CommonStrategy")         # Runs commonly used indicators
```

## 📊 Features

- **141 Technical Indicators & Utilities** across 9 categories (Candles, Momentum, Overlap, Trend, Volume, etc.)
- **62 TA Lib Candlestick Patterns** for comprehensive pattern recognition
- **203 Total Indicators & Patterns** - the most comprehensive Python TA library
- **Dynamic Category Discovery** - automatically detects all available indicators from the filesystem
- **Strategy System** with multiprocessing support for bulk indicator processing
- **Pandas DataFrame Extension** for seamless integration (`df.ta.indicator()`)
- **TA Lib Integration** - automatically uses TA Lib versions when available
- **Vectorbt Integration** - compatible with popular backtesting framework
- **Custom Indicators** - easily create and chain your own indicators

## 📚 Documentation

**Complete documentation is available at:** 🔗 **[https://xgboosted.github.io/pandas-ta-classic/](https://xgboosted.github.io/pandas-ta-classic/)**

### Quick Links
- 📖 [**Usage Guide**](https://xgboosted.github.io/pandas-ta-classic/usage.html) - Programming conventions and basic usage
- 🏗️ [**Strategy System**](https://xgboosted.github.io/pandas-ta-classic/strategies.html) - Multiprocessing and bulk indicator processing  
- 📊 [**Indicators Reference**](https://xgboosted.github.io/pandas-ta-classic/indicators.html) - Complete list of all 141 indicators & 62 patterns
- 🔧 [**DataFrame API**](https://xgboosted.github.io/pandas-ta-classic/dataframe_api.html) - Properties and methods reference
- 📈 [**Performance Metrics**](https://xgboosted.github.io/pandas-ta-classic/performance.html) - Backtesting and performance analysis
- 💡 [**Examples**](https://github.com/xgboosted/pandas-ta-classic/tree/main/examples) - Jupyter notebooks and code examples

## 🐍 Python Version Support

**Pandas TA Classic** follows a **rolling support policy** for the latest stable Python version plus 4 preceding minor versions.

> **Note:** Python version support is **dynamically managed** via CI/CD workflows. When new Python versions are released, the library automatically updates to support the latest 5 minor versions. Check the [CI workflow](.github/workflows/ci.yml) `LATEST_PYTHON_VERSION` for the current configuration.

**Note:** _TA Lib_ installation enables all candlestick patterns:
- Using `uv`: `uv pip install TA-Lib`
- Using `pip`: `pip install TA-Lib`

## 🤝 Contributing

We welcome contributions! Please see our [contributing guidelines](https://github.com/xgboosted/pandas-ta-classic/blob/main/CONTRIBUTING.md) and [issues page](https://github.com/xgboosted/pandas-ta-classic/issues).

### Reporting Issues
- Check [existing issues](https://github.com/xgboosted/pandas-ta-classic/issues) first
- Provide reproducible code examples  
- Include relevant error messages and data samples

## 📋 Changelog

For detailed information about changes, improvements, and new features, please see the [CHANGELOG.md](CHANGELOG.md) file.

## 🔗 Sources

[Original TA-LIB](http://ta-lib.org/) | [TradingView](http://www.tradingview.com) | [Sierra Chart](https://search.sierrachart.com/?Query=indicators&submitted=true) | [MQL5](https://www.mql5.com) | [FM Labs](https://www.fmlabs.com/reference/default.htm) | [Pro Real Code](https://www.prorealcode.com/prorealtime-indicators) | [User 42](https://user42.tuxfamily.org/chart/manual/index.html)

## ❤️ Support

If you find this library helpful, please consider:

[![Sponsor](https://img.shields.io/static/v1?label=Sponsor&message=%E2%9D%A4&logo=GitHub&color=%23fe8e86)](https://github.com/sponsors/xgboosted)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.