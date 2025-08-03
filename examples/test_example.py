#!/usr/bin/env python3
"""
Test script for example.ipynb content without Jupyter magic commands
"""
import datetime as dt
import random as rnd

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
import mplfinance as mpf

# Optional import for AlphaVantage API
try:
    from alphaVantageAPI.alphavantage import AlphaVantage
except ImportError:
    print("[!] alphaVantageAPI not available. Install with: pip install alphaVantage-api")
    AlphaVantage = None

import pandas_ta_classic as ta

from watchlist import colors # Import from our local watchlist module

print(f"\nPandas TA Classic v{ta.version}\nTo install the Latest Version:\n$ pip install pandas-ta-classic\n")

# Test basic functionality
try:
    # Simple DataFrame test
    df_test = pd.DataFrame({'close': [1,2,3,4,5]})
    print("Basic pandas test: OK")
    
    # Test pandas-ta-classic functionality
    df_test.ta.sma(length=2, append=True)
    print("Basic pandas-ta-classic test: OK")
    
    # Test watchlist colors function
    test_colors = colors("GrRd")
    print(f"Watchlist colors test: OK - Got {test_colors}")
    
except Exception as e:
    print(f"Error in basic tests: {e}")
    raise

print("All basic tests passed!")